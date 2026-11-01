"use strict";
// DragonScope exhibit. See docs/PROTOCOL.md.
//
// Each `state` message replaces the whole view, and no earlier copy is kept. Live
// frames hold the byte being computed, so they are dropped when that state arrives.

const $ = (id) => document.getElementById(id);

const FIELD_LABELS = { card: "Card", nic: "NIC", name: "Name", branch: "Branch" };
const KIND_WORDS = { card: "card", nic: "NIC", number: "number" };

let socket = null;
let guide = null;
let state = null;
let forgetResult = null; // Holds measurements only, never a value.
let storyActive = false; // True from a Forget until the visitor shares something new.
let busy = false;
let clearInput = false;
let activeField = "card";
let lastHeadline = "";
let playback = null; // The exchange being computed right now, shown as it happens.
let recall = null; // Counts only: how many written characters depended on memory.
let model = null; // The model's real size and structure, from the server.

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function button(className, text, onclick) {
  const node = el("button", className, text);
  node.type = "button";
  node.disabled = busy;
  node.onclick = onclick;
  return node;
}

function send(message) {
  if (!socket || socket.readyState !== WebSocket.OPEN || busy) return;
  busy = true;
  showNote();
  socket.send(JSON.stringify(message));
  render();
}

function connect() {
  socket = new WebSocket(`ws://${location.host}/ws`);
  socket.onopen = () => { $("connection").hidden = true; };
  socket.onclose = () => {
    $("connection").hidden = false;
    playback = null;
    $("readout").hidden = true;
    busy = false;
    setTimeout(connect, 1000);
  };
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "hello") {
      guide = message.guide;
      model = message.model;
      brain.build(model);
      renderHud();
      applyState(message.state);
      renderFields();
    } else if (message.type === "frame") {
      onFrame(message);
    } else if (message.type === "state") {
      applyState(message);
    } else if (message.type === "forget_result") {
      forgetResult = message;
      storyActive = true;
      recall = null;
    } else if (message.type === "error") {
      busy = false;
      clearInput = false;
      showNote(message.message);
      render();
    }
  };
}

function applyState(next) {
  if (playback && next.last_answer) {
    recall = {
      field: next.last_answer.field,
      count: playback.written.filter((depends) => depends > 0.5).length,
      total: playback.written.length,
    };
  }
  playback = null;
  $("readout").hidden = true;
  state = next;
  if (!state.turns.length) {
    forgetResult = null;
    storyActive = false;
    recall = null;
    activeField = "card";
  }
  if (clearInput) {
    clearInput = false;
    const open = guide.fields.find((field) => !state.stated.includes(field));
    if (open && state.stated.includes(activeField)) activeField = open;
    renderFields();
  }
  busy = false;
  brain.showMemory(state.memory);
  renderHud();
  render();
}

function render() {
  if (!guide || !state) return;
  renderStory();
  renderTranscript();
  renderChoices();
  renderProof();
  const { position, position_limit: limit } = state.memory;
  $("session-fill").style.width = `${Math.min(100, (position / limit) * 100)}%`;
  $("reset").disabled = busy;
  $("speed").disabled = busy;
  $("speed").textContent = state.full_speed ? "Speed: full" : "Speed: paced to watch";
  $("say-send").disabled = busy || state.stated.includes(activeField);
}

// The model's work, shown as each byte is computed

function onFrame(frame) {
  if (!playback) {
    playback = {
      user: "",
      reply: "",
      written: [],
      phase: "read",
      question: frame.question,
      recalled: false,
      depends: 0,
    };
    busy = true;
    render();
  }
  const run = playback;
  run.phase = frame.phase;
  brain.fireFrame(frame);
  renderHud(frame);
  if (frame.phase === "read") {
    if (frame.char) {
      run.user += frame.char;
      renderTranscript();
    }
    renderStory();
    return;
  }
  if (frame.char) {
    run.reply += frame.char;
    run.written.push(frame.depends);
    run.depends = frame.depends;
    if (frame.depends > 0.5) run.recalled = true;
  }
  renderStory();
  renderTranscript();
  renderReadout();
}

// The model's real numbers, and how many of its neurons are firing right now.
function renderHud(frame) {
  const hud = $("hud");
  hud.replaceChildren();
  const add = (value, label, className) => {
    const stat = el("div", `hud-stat${className ? ` ${className}` : ""}`);
    stat.append(el("b", "", value), el("span", "", label));
    hud.append(stat);
  };
  const mean = (values) => Math.round(values.reduce((sum, value) => sum + value, 0) / values.length);
  if (frame) {
    const hot = frame.depends > 0.5;
    add(mean(frame.counts).toLocaleString("en-US"), `of ${model.neurons.toLocaleString("en-US")} neurons firing`, `live${hot ? " hot" : ""}`);
    add(mean(frame.passed).toLocaleString("en-US"), "passing their signal on", `live${hot ? " hot" : ""}`);
    add(`${Math.round(frame.ms)} ms`, "to compute this character");
  } else {
    add(model.neurons.toLocaleString("en-US"), "real neurons");
    add(`${(model.parameters / 1e6).toFixed(1)}M`, "learned weights");
    add(`${model.rounds} × ${model.heads}`, "rounds per character × heads");
  }
  add("0", "cloud calls: offline, on CPU");
}

function renderReadout() {
  const run = playback;
  if (!run || !run.question) return;
  $("readout").hidden = false;
  const share = Math.round(run.depends * 100);
  $("readout-label").textContent = state.findings.length || run.depends > 0
    ? `This character depends on your private data: ${share}%`
    : "No private data in memory to draw on";
  $("readout-fill").style.width = `${share}%`;
}

// The one sentence that tells the visitor what is happening

function groupFindings() {
  const groups = new Map();
  state.findings.forEach((finding) => {
    const key = `${finding.kind}:${finding.last4}`;
    if (!groups.has(key)) {
      groups.set(key, { kind: finding.kind, last4: finding.last4, id: finding.id, source: finding.source });
    }
  });
  return [...groups.values()];
}

function story() {
  const groups = groupFindings();
  const type = forgetResult && forgetResult.log_entry.type;
  const answer = state.last_answer;

  if (playback) {
    if (playback.phase === "read") {
      return { step: 0, text: playback.question ? "Reading your question…" : "Storing what you said…" };
    }
    if (playback.recalled) return { step: 0, text: "Recalling from memory…", alert: true };
    return { step: 0, text: playback.question ? "Answering…" : "Replying…" };
  }
  if (busy) return { step: 0, text: "Working…" };

  if (storyActive && answer && answer.field === type) {
    return answer.refused
      ? {
        step: 5,
        text: "It no longer knows.",
        sub: "Same question, same brain. There is nothing left in memory to recall.",
        restart: true,
      }
      : { step: 5, text: "It did not refuse.", alert: true, sub: "This reply needs a closer look.", restart: true };
  }
  if (storyActive && !answer) {
    return {
      step: 4,
      text: "Forgotten.",
      sub: "The red is gone. Ask the same question and watch what happens.",
      askAgain: guide.fields.includes(type) ? type : null,
    };
  }

  const held = groups.find((group) => answer && group.kind === answer.field && !answer.refused);
  if (held) {
    const counted = recall && recall.field === answer.field && recall.total;
    return {
      step: 3,
      text: `It recalled your ${KIND_WORDS[held.kind]} from memory.`,
      alert: true,
      sub: counted
        ? `Without that memory, it would likely not have written ${recall.count} of these ${recall.total} characters.`
        : "Now erase it.",
    };
  }
  if (groups.length) {
    const first = groups.find((group) => group.source === "user") || groups[0];
    return {
      step: 2,
      text: groups.length === 1
        ? `It remembers your ${KIND_WORDS[first.kind]} ending ${first.last4}.`
        : `It remembers ${groups.length} sensitive details.`,
      alert: true,
      sub: "The red neurons hold it. Ask for it back and watch them fire.",
      ask: guide.fields.includes(first.kind) ? first.kind : null,
    };
  }
  if (state.turns.length) {
    return { step: 1, text: "Nothing sensitive in memory.", sub: "Now tell it a card number or NIC." };
  }
  return { step: 1, text: "Tell it something private.", sub: "Type a fake card number below, and watch the brain store it." };
}

function renderStory() {
  const now = story();
  const headline = $("headline");
  if (now.text !== lastHeadline) {
    headline.classList.remove("enter");
    void headline.offsetWidth; // Restart the animation.
    headline.classList.add("enter");
    lastHeadline = now.text;
  }
  headline.textContent = now.text;
  headline.classList.toggle("alert", Boolean(now.alert));
  $("subline").textContent = now.sub || "";

  if (now.step) {
    [...$("steps").children].forEach((item, index) => {
      item.classList.toggle("done", index + 1 < now.step);
      item.classList.toggle("current", index + 1 === now.step);
    });
  }
  $("entry").classList.toggle("invite", now.step === 1);

  const actions = $("actions");
  actions.replaceChildren();
  if (playback) return;
  if (now.ask) {
    const ask = button("act", guide.questions[now.ask], () => send({ type: "ask", field: now.ask }));
    ask.id = "ask-now";
    actions.append(ask);
  }
  if (now.askAgain) {
    const again = button("act", "Ask it again", () => send({ type: "ask", field: now.askAgain }));
    again.id = "ask-again";
    actions.append(again);
  }
  if (now.step === 3 || now.step === 4) {
    groupFindings().forEach((group, _, all) => {
      const label = all.length === 1 ? "Forget" : `Forget ${KIND_WORDS[group.kind]} ${group.last4}`;
      actions.append(button("act forget", busy ? "Erasing…" : label, () => send({ type: "forget", finding: group.id })));
    });
  }
  if (now.restart) {
    actions.append(button("act", "Start over", () => { clearInput = true; send({ type: "reset" }); }));
  }
}

// Conversation

function renderTurn(turn) {
  const node = el("p", `turn ${turn.role}`);
  turn.segments.forEach((segment) => {
    if (segment.forgotten) node.append(el("span", "forgotten", "forgotten"));
    else if (segment.finding) node.append(el("span", "flagged", segment.text));
    else node.append(document.createTextNode(segment.text));
  });
  return node;
}

// While a reply is being written, each character's colour shows how much it
// depends on private data, as measured for that character.
function renderTyping(text, depends) {
  const node = el("p", "turn assistant");
  [...text].forEach((character, index) => {
    const span = el("span", "", character);
    if (depends[index] > 0.05) {
      span.style.color = `color-mix(in srgb, #ff4b3e ${Math.round(depends[index] * 100)}%, #f2efe9)`;
    }
    node.append(span);
  });
  node.append(el("span", "caret"));
  return node;
}

function renderTranscript() {
  const box = $("transcript");
  box.replaceChildren();
  state.turns.forEach((turn) => box.append(renderTurn(turn)));
  if (playback) {
    box.append(el("p", "turn user", playback.user));
    if (playback.phase === "write") box.append(renderTyping(playback.reply, playback.written));
  }
  if (busy && !playback) box.append(el("p", "turn user", "…"));
  box.scrollTop = box.scrollHeight;
}

// Guided inputs

function renderChoices() {
  const tabs = $("tabs");
  tabs.replaceChildren();
  guide.fields.forEach((field) => {
    const tab = button(`link${field === activeField ? " current" : ""}`, FIELD_LABELS[field], () => {
      activeField = field;
      showNote();
      renderFields();
      render();
    });
    tab.disabled = busy || state.stated.includes(field);
    tab.setAttribute("aria-pressed", String(field === activeField));
    tabs.append(tab);
  });
  const asks = $("asks");
  asks.replaceChildren();
  guide.fields.forEach((field) => {
    const ask = button("link", FIELD_LABELS[field], () => send({ type: "ask", field }));
    ask.title = guide.questions[field];
    ask.dataset.ask = field;
    asks.append(ask);
  });
}

function dropdown(id, label, options) {
  const node = el("select");
  node.id = id;
  node.setAttribute("aria-label", label);
  options.forEach((option) => node.append(new Option(option, option)));
  return node;
}

function renderFields() {
  const box = $("say-fields");
  box.replaceChildren();
  if (activeField === "name") {
    box.append(dropdown("say-first", "First name", guide.first_names), dropdown("say-last", "Last name", guide.last_names));
  } else if (activeField === "branch") {
    box.append(dropdown("say-branch", "Branch", guide.branches));
  } else {
    const input = el("input");
    input.id = "say-value";
    input.spellcheck = false;
    input.setAttribute("aria-label", activeField === "card" ? "Card number" : "NIC");
    input.placeholder = activeField === "card" ? "Card number" : "NIC";
    box.append(input);
  }
  showNote();
}

function showNote(error) {
  const note = $("form-note");
  note.replaceChildren();
  note.classList.toggle("error", Boolean(error));
  if (error) {
    note.textContent = error;
  } else if (activeField === "card" || activeField === "nic") {
    note.append(document.createTextNode("Use the printed fake number, or "));
    const fill = el("button", "link", "make one up");
    fill.type = "button";
    fill.id = "sample";
    fill.onclick = () => {
      const input = $("say-value");
      if (input) input.value = activeField === "card" ? fakeCard() : fakeNic();
    };
    note.append(fill, document.createTextNode("."));
  }
}

function currentValue() {
  if (activeField === "name") return `${$("say-first").value} ${$("say-last").value}`;
  if (activeField === "branch") return $("say-branch").value;
  return $("say-value").value;
}

function randomDigits(count) {
  return Array.from(crypto.getRandomValues(new Uint8Array(count)), (value) => value % 10).join("");
}

function fakeCard() {
  const body = (Math.random() < 0.5 ? "4" : "5") + randomDigits(14);
  let total = 0;
  [...body].reverse().forEach((character, index) => {
    let digit = Number(character);
    if (index % 2 === 0) { digit *= 2; if (digit > 9) digit -= 9; }
    total += digit;
  });
  return (body + String((10 - total % 10) % 10)).match(/.{4}/g).join(" ");
}

function fakeNic() {
  return Math.random() < 0.5
    ? randomDigits(9) + (Math.random() < 0.5 ? "V" : "X")
    : randomDigits(12);
}

// Proof, measured on each Forget

function percent(value) { return `${(value * 100).toFixed(1)}%`; }

function stat(value, label, bad) {
  const node = el("div", "stat");
  node.append(el("b", bad ? "bad" : "", value), el("span", "", label));
  return node;
}

function renderProof() {
  const box = $("proof");
  box.replaceChildren();
  if (!forgetResult) return;
  const checks = forgetResult.checks;
  const traces = (checks.value_in_memory_bytes ? 1 : 0) + checks.ledger_entries_left + (checks.pending_logits ? 1 : 0);
  const copies = forgetResult.copies_erased;
  const stats = el("div", "stats");
  stats.append(
    stat(percent(forgetResult.memory_change), "memory changed"),
    stat(
      `${percent(forgetResult.residual_after_subtract)} → ${percent(forgetResult.residual_after_rebuild)}`,
      "from never seeing it: subtract, then rebuild",
      forgetResult.residual_after_rebuild > 0.0005,
    ),
    stat(String(traces), `traces left, after erasing ${copies} ${copies === 1 ? "copy" : "copies"}`, traces > 0),
  );
  box.append(stats);
  const log = el("div", "log");
  state.deletion_log.forEach((entry) => {
    const time = new Date(entry.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    log.append(el("div", "", `Deleted: ${entry.type} ${entry.last4} at ${time}`));
  });
  box.append(log);
  box.append(el("p", "proof-note", "Subtracting leaves a trace, so memory is rebuilt from the surviving text. Session memory in a prototype, not a security proof."));
}

// Wiring

$("say-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!guide || busy || state.stated.includes(activeField)) return;
  clearInput = true;
  storyActive = false;
  send({ type: "say", field: activeField, value: currentValue() });
});

$("speed").addEventListener("click", () => send({ type: "speed", full: !state.full_speed }));

$("reset").addEventListener("click", () => {
  clearInput = true;
  send({ type: "reset" });
});

connect();
