"use strict";
// The DragonScope dashboard. It works with any session and model size.
//
// Each `state` message replaces the view, and live frames are dropped when it
// arrives, so a forgotten value leaves the page with the next state.

const $ = (id) => document.getElementById(id);
let socket = null;
let model = null;
let state = null;
let busy = false;
let live = null; // Text being computed right now: { read, written, depends }.
let writtenFrom = null; // Where the latest generated text starts, to mark it.

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function send(message) {
  if (!socket || socket.readyState !== WebSocket.OPEN || busy) return;
  busy = true;
  $("error").textContent = "";
  socket.send(JSON.stringify(message));
  renderControls();
}

function connect() {
  socket = new WebSocket(`ws://${location.host}/ws`);
  socket.onopen = () => { $("connection").hidden = true; };
  socket.onclose = () => {
    $("connection").hidden = false;
    live = null;
    busy = false;
    setTimeout(connect, 1000);
  };
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "hello") {
      model = message.model;
      brain.build(model);
      applyState(message.state);
    } else if (message.type === "frame") {
      onFrame(message);
    } else if (message.type === "state") {
      applyState(message);
    } else if (message.type === "error") {
      busy = false;
      live = null;
      $("error").textContent = message.message;
      renderControls();
      renderHud();
    }
  };
}

function applyState(next) {
  state = next;
  live = null;
  busy = false;
  $("readout").hidden = true;
  brain.showMemory(state.memory);
  renderText();
  renderControls();
  renderHud();
  renderResults();
  const { position, position_limit: limit } = state.memory;
  $("session-fill").style.width = `${Math.min(100, (position / limit) * 100)}%`;
}

function onFrame(frame) {
  if (!live) {
    live = { text: "", written: [] };
    busy = true;
    renderControls();
  }
  live.text += frame.char;
  if (frame.phase === "write") live.written.push(frame.depends);
  brain.fireFrame(frame);
  renderHud(frame);
  renderText();
  if (frame.change.length && frame.phase === "write") {
    const share = Math.round(frame.depends * 100);
    $("readout").hidden = false;
    $("readout-label").textContent = `This byte depends on watched text: ${share}%`;
    $("readout-fill").style.width = `${share}%`;
  }
}

// The text in memory, keeping each character's position so it can be selected

function isWatched(position) {
  return state.watched.some(([start, end]) => position >= start && position < end);
}

function renderText() {
  if (!state) return;
  const box = $("text");
  box.replaceChildren();
  state.runs.forEach((run, index) => {
    if (index > 0 || run.start > 0) box.append(el("span", "gap", "forgotten"));
    let segment = null;
    let segmentWatched = null;
    [...run.text].forEach((character, offset) => {
      const position = run.start + offset;
      const watched = isWatched(position);
      if (segment === null || watched !== segmentWatched) {
        segment = el("span", watched ? "watched" : "");
        segment.dataset.start = String(position);
        segmentWatched = watched;
        box.append(segment);
      }
      segment.append(document.createTextNode(character));
    });
  });
  if (live) {
    const node = el("span", "latest", live.text);
    box.append(node);
  }
  box.scrollTop = box.scrollHeight;
}

function selectedText() {
  const selection = window.getSelection();
  if (!selection || selection.isCollapsed) return "";
  const box = $("text");
  if (!box.contains(selection.anchorNode) || !box.contains(selection.focusNode)) return "";
  return selection.toString();
}

function renderControls() {
  ["read", "watch", "forget", "rebuild", "reset", "run-report"].forEach((id) => { $(id).disabled = busy; });
  $("generate").disabled = busy || !state || !state.can_generate;
}

function renderHud(frame) {
  const hud = $("hud");
  hud.replaceChildren();
  const add = (value, label, className) => {
    const stat = el("div", `hud-stat${className ? ` ${className}` : ""}`);
    stat.append(el("b", "", value), el("span", "", label));
    hud.append(stat);
  };
  if (!model || !state) return;
  const mean = (values) => Math.round(values.reduce((sum, value) => sum + value, 0) / values.length);
  const number = (value) => value.toLocaleString("en-US");
  if (frame) {
    const hot = frame.depends > 0.5;
    add(number(mean(frame.counts)), `of ${number(model.neurons)} neurons firing`, `live${hot ? " hot" : ""}`);
    add(number(mean(frame.passed)), "passing their signal on", `live${hot ? " hot" : ""}`);
    add(`${Math.round(frame.ms)} ms`, "to compute this byte");
  } else {
    add(number(model.neurons), `neurons, in ${model.heads} heads`);
    add(number(state.memory.bytes_stored), "bytes in memory");
    add(`${state.memory.rho_mb} MB`, "memory state, fixed size");
  }
  add(`${state.memory.ledger_mb} MB`, "ledger, grows per byte");
}

function percent(value) { return value === null || value === undefined ? "n/a" : `${(value * 100).toFixed(1)}%`; }

function renderResults() {
  const box = $("results");
  box.replaceChildren();
  const edit = state.last_edit;
  if (edit) {
    const stats = el("div", "stats");
    const add = (value, label) => {
      const stat = el("div", "stat");
      stat.append(el("b", "", value), el("span", "", label));
      stats.append(stat);
    };
    add(`${edit.ms} ms`, `to ${edit.method}, ${edit.bytes_erased} bytes in ${edit.copies} ${edit.copies === 1 ? "copy" : "copies"}`);
    add(percent(edit.residual), "distance from never reading them");
    add(edit.residual_per_round.map((value) => percent(value)).join(" "), "that distance, by round");
    box.append(stats);
  }
  const report = state.report;
  if (report) {
    box.append(el("p", `small ${report.passed ? "verdict-pass" : "verdict-fail"}`, `Forgetting report: ${report.passed ? "PASS" : "FAIL"}`));
    const table = el("table", "results-table");
    const head = el("thead");
    const headRow = el("tr");
    ["", "Before", "subtract", "rebuild"].forEach((text) => headRow.append(el("th", "", text)));
    head.append(headRow);
    const body = el("tbody");
    const row = (label, values) => {
      const tr = el("tr");
      tr.append(el("th", "", label));
      values.forEach((value) => tr.append(el("td", "", value)));
      body.append(tr);
    };
    const conditions = ["before", "subtract", "rebuild"];
    row("Recall of forgotten text", conditions.map((c) => percent(report.recall[c].forgotten)));
    row("Recall of kept text", conditions.map((c) => percent(report.recall[c].kept)));
    row("Time to forget", ["n/a", `${(report.seconds.subtract * 1000).toFixed(1)} ms`, `${(report.seconds.rebuild * 1000).toFixed(1)} ms`]);
    row("Distance from never reading it", ["n/a", percent(report.residual.subtract), percent(report.residual.rebuild)]);
    table.append(head, body);
    box.append(table);
    Object.entries(report.checks).filter(([, ok]) => !ok).forEach(([name]) => {
      box.append(el("p", "small verdict-fail", `Failed: ${name.replaceAll("`", "")}`));
    });
    box.append(el("p", "proof-note", "This covers these probes on this model. It is evidence, not a proof."));
  }
}

// Wiring

document.addEventListener("selectionchange", () => {
  const text = selectedText();
  if (text) $("target").value = text;
});
$("read").addEventListener("click", () => {
  const text = $("input").value;
  send({ type: "read", text });
  $("input").value = "";
});
$("input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) $("read").click();
});
$("generate").addEventListener("click", () => send({ type: "generate", count: Number($("count").value) || 70 }));
$("watch").addEventListener("click", () => send({ type: "watch", text: $("target").value }));
$("forget").addEventListener("click", () => {
  const text = $("target").value;
  send({ type: "forget", text, method: $("method").value });
  // Forgetting must leave no copy on the page, including in the report's form.
  $("target").value = "";
  if ($("r-forget").value === text) $("r-forget").value = "";
  window.getSelection().removeAllRanges();
});
$("rebuild").addEventListener("click", () => send({ type: "rebuild" }));
$("reset").addEventListener("click", () => send({ type: "reset" }));
$("run-report").addEventListener("click", () => {
  const forget = $("r-forget").value;
  const unescape = (text) => text.replaceAll("\\n", "\n");
  const probes = [{ prompt: unescape($("r-prompt").value), expect: forget, about: "forgotten" }];
  if ($("r-kept-prompt").value) {
    probes.push({ prompt: unescape($("r-kept-prompt").value), expect: $("r-kept").value, about: "kept" });
  }
  send({ type: "report", forget, probes });
});

connect();
