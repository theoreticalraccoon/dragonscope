"use strict";
// DragonScope exhibit. See docs/PROTOCOL.md.
//
// Each `state` message replaces the whole view and no earlier copy is kept, so a
// forgotten value leaves the page as soon as the next state arrives.

const $ = (id) => document.getElementById(id);

const FIELD_LABELS = { card: "Card", nic: "NIC", name: "Name", branch: "Branch" };
const KIND_WORDS = { card: "card", nic: "NIC", number: "number" };
const FAKE_NOTE = "Use the printed fake number, or ";

let socket = null;
let guide = null;
let state = null;
let forgetResult = null; // Holds measurements only, never a value.
let storyActive = false; // True from a Forget until the visitor shares something new.
let busy = false;
let clearInput = false;
let activeField = "card";
let lastHeadline = "";

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
    busy = false;
    setTimeout(connect, 1000);
  };
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "hello") {
      guide = message.guide;
      applyState(message.state);
      renderFields();
    } else if (message.type === "state") {
      applyState(message);
    } else if (message.type === "forget_result") {
      forgetResult = message;
      storyActive = true;
    } else if (message.type === "error") {
      busy = false;
      clearInput = false;
      showNote(message.message);
      render();
    }
  };
}

function applyState(next) {
  state = next;
  busy = false;
  if (!state.turns.length) {
    forgetResult = null;
    storyActive = false;
    activeField = "card";
  }
  if (clearInput) {
    clearInput = false;
    const open = guide.fields.find((field) => !state.stated.includes(field));
    if (open && state.stated.includes(activeField)) activeField = open;
    renderFields();
  }
  render();
  memoryMap.update(state.memory.neurons);
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
  $("say-send").disabled = busy || state.stated.includes(activeField);
}

// The one sentence that tells the visitor what is happening

function groupFindings() {
  const groups = new Map();
  state.findings.forEach((finding) => {
    const key = `${finding.kind}:${finding.last4}`;
    if (!groups.has(key)) groups.set(key, { kind: finding.kind, last4: finding.last4, id: finding.id });
  });
  return [...groups.values()];
}

function renderStory() {
  const groups = groupFindings();
  const type = forgetResult && forgetResult.log_entry.type;
  const answer = state.last_answer;
  const reasked = storyActive && answer && answer.field === type;
  let text;
  let alert = false;

  if (reasked) {
    text = answer.refused ? "It no longer knows." : "It did not refuse.";
    alert = !answer.refused;
  } else if (storyActive && !answer) {
    text = "Forgotten.";
  } else if (groups.length === 1) {
    text = `It remembers your ${KIND_WORDS[groups[0].kind]} ending ${groups[0].last4}.`;
    alert = true;
  } else if (groups.length > 1) {
    text = `It remembers ${groups.length} sensitive details.`;
    alert = true;
  } else if (state.turns.length) {
    text = "Nothing sensitive in memory.";
  } else {
    text = "Tell it something private.";
  }

  const headline = $("headline");
  if (text !== lastHeadline) {
    headline.classList.remove("enter");
    void headline.offsetWidth; // Restart the animation.
    headline.classList.add("enter");
    lastHeadline = text;
  }
  headline.textContent = text;
  headline.classList.toggle("alert", alert);

  const actions = $("actions");
  actions.replaceChildren();
  if (storyActive && !answer && guide.fields.includes(type)) {
    actions.append(button("act", "Ask it again", () => send({ type: "ask", field: type })));
  }
  groups.forEach((group) => {
    const label = groups.length === 1 ? "Forget" : `Forget ${KIND_WORDS[group.kind]} ${group.last4}`;
    const forget = button("act forget", busy ? "Working…" : label, () => send({ type: "forget", finding: group.id }));
    actions.append(forget);
  });
}

// Conversation

function renderTranscript() {
  const box = $("transcript");
  box.replaceChildren();
  state.turns.forEach((turn) => {
    const node = el("p", `turn ${turn.role}`);
    turn.segments.forEach((segment) => {
      if (segment.forgotten) node.append(el("span", "forgotten", "forgotten"));
      else if (segment.finding) node.append(el("span", "flagged", segment.text));
      else node.append(document.createTextNode(segment.text));
    });
    box.append(node);
  });
  if (busy) box.append(el("p", "turn user", "…"));
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
    note.append(document.createTextNode(FAKE_NOTE));
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
  const stats = el("div", "stats");
  stats.append(
    stat(percent(forgetResult.memory_change), "memory changed"),
    stat(
      `${percent(forgetResult.residual_after_subtract)} → ${percent(forgetResult.residual_after_rebuild)}`,
      "from never seeing it: subtract, then rebuild",
      forgetResult.residual_after_rebuild > 0.0005,
    ),
    stat(String(traces), `traces left, after erasing ${forgetResult.copies_erased} ${forgetResult.copies_erased === 1 ? "copy" : "copies"}`, traces > 0),
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

// The scope: rings are layers and quadrants are heads. A neuron keeps its place
// between updates, and its size and colour animate.

const memoryMap = (() => {
  const canvas = $("map");
  const context = canvas.getContext("2d");
  const LAYERS = 4;
  const HEADS = 4;
  const WHITE = "#f2efe9";
  const RED = "#ff4b3e";
  const RING = "#262624";
  const colour = d3.interpolateRgb(WHITE, RED);
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const nodes = new Map();
  let width = 0;
  let height = 0;
  let timer = null;

  const fraction = (value) => value - Math.floor(value);
  const radiusOf = () => Math.max(10, Math.min(width, height) / 2 - 10);
  const ringStart = (layer) => radiusOf() * (0.2 + layer * 0.2);

  function place(node) {
    const span = Math.PI / 2;
    const gap = 0.08;
    const angle = node.head * span + gap + fraction(node.neuron * 0.6180339887) * (span - gap * 2);
    const distance = ringStart(node.layer) + radiusOf() * 0.2 * (0.1 + 0.8 * fraction(node.neuron * 0.7548776662));
    return {
      x: width / 2 + Math.cos(angle) * distance,
      y: height / 2 + Math.sin(angle) * distance,
    };
  }

  function drawRings() {
    context.strokeStyle = RING;
    context.lineWidth = 1;
    for (let layer = 0; layer <= LAYERS; layer += 1) {
      context.beginPath();
      context.arc(width / 2, height / 2, ringStart(layer), 0, Math.PI * 2);
      context.stroke();
    }
    for (let head = 0; head < HEADS; head += 1) {
      const angle = head * (Math.PI / 2);
      context.beginPath();
      context.moveTo(width / 2 + Math.cos(angle) * ringStart(0), height / 2 + Math.sin(angle) * ringStart(0));
      context.lineTo(width / 2 + Math.cos(angle) * ringStart(LAYERS), height / 2 + Math.sin(angle) * ringStart(LAYERS));
      context.stroke();
    }
  }

  function draw() {
    context.clearRect(0, 0, width, height);
    drawRings();
    const ordered = [...nodes.values()].sort((a, b) => a.share - b.share);
    const hot = ordered.filter((node) => node.share > 0.3 && node.radius > 0.5).slice(-30);

    // Threads from the centre to the neurons holding the most flagged data.
    context.strokeStyle = RED;
    context.lineWidth = 1;
    hot.forEach((node) => {
      const { x, y } = place(node);
      context.globalAlpha = 0.1 + node.share * 0.35;
      context.beginPath();
      context.moveTo(width / 2, height / 2);
      context.lineTo(x, y);
      context.stroke();
    });

    ordered.forEach((node) => {
      if (node.radius <= 0.1) return;
      const { x, y } = place(node);
      const heat = node.share; // Unscaled, so red shows the true share.
      context.fillStyle = colour(heat);
      context.globalAlpha = 0.45 + heat * 0.55;
      context.beginPath();
      context.arc(x, y, node.radius, 0, Math.PI * 2);
      context.fill();
    });
    context.globalAlpha = 1;

    context.fillStyle = hot.length ? RED : WHITE;
    context.beginPath();
    context.arc(width / 2, height / 2, 3, 0, Math.PI * 2);
    context.fill();
  }

  function resize() {
    const box = canvas.parentElement.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    width = box.width;
    height = box.height;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    draw();
  }

  function update(neurons) {
    const maximum = d3.max(neurons, (neuron) => neuron.strength) || 1;
    const scale = d3.scaleSqrt().domain([0, maximum]).range([1, 7]);
    const seen = new Set();
    neurons.forEach((neuron) => {
      const key = `${neuron.layer}.${neuron.head}.${neuron.neuron}`;
      seen.add(key);
      const node = nodes.get(key) || { ...neuron, radius: 0, share: 0 };
      node.toRadius = scale(neuron.strength);
      node.toShare = neuron.flagged_share;
      nodes.set(key, node);
    });
    nodes.forEach((node, key) => {
      node.fromRadius = node.radius;
      node.fromShare = node.share;
      if (!seen.has(key)) {
        node.toRadius = 0;
        node.toShare = 0;
      }
    });
    const finish = () => {
      nodes.forEach((node, key) => {
        node.radius = node.toRadius;
        node.share = node.toShare;
        if (!seen.has(key)) nodes.delete(key);
      });
      draw();
    };
    if (timer) { timer.stop(); timer = null; }
    if (reduceMotion) { finish(); return; }
    timer = d3.timer((elapsed) => {
      const t = d3.easeCubicInOut(Math.min(1, elapsed / 900));
      nodes.forEach((node) => {
        node.radius = node.fromRadius + (node.toRadius - node.fromRadius) * t;
        node.share = node.fromShare + (node.toShare - node.fromShare) * t;
      });
      draw();
      if (elapsed >= 900) { timer.stop(); timer = null; finish(); }
    });
  }

  new ResizeObserver(resize).observe(canvas.parentElement);
  return { update };
})();

// Wiring

$("say-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!guide || busy || state.stated.includes(activeField)) return;
  clearInput = true;
  storyActive = false;
  send({ type: "say", field: activeField, value: currentValue() });
});

$("reset").addEventListener("click", () => {
  clearInput = true;
  send({ type: "reset" });
});

connect();
