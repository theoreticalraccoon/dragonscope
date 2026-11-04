"use strict";
// Review console for python -m dragonscope.gateway. Everything is drawn with
// textContent, so case text can never run as markup. Nothing is stored locally.

const $ = (id) => document.getElementById(id);
const CODE = /\b(?:PERSON|CARD|NIC|PHONE|EMAIL|RECORD|NUMBER|PLACE)_[A-Z]{4}\b/g;

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "text") node.textContent = value;
    else if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) if (child) node.append(child);
  return node;
}

function chip(text, kind = "") {
  return el("span", { class: `chip ${kind}`, text });
}

async function post(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Something went wrong.");
  return data;
}

function showError(message) {
  $("error").textContent = message || "";
}

// Highlights placeholders in the text sent out.
function withCodes(text) {
  const pre = el("pre");
  let last = 0;
  for (const match of text.matchAll(CODE)) {
    pre.append(text.slice(last, match.index), el("mark", { text: match[0] }));
    last = match.index + match[0].length;
  }
  pre.append(text.slice(last));
  return pre;
}

const STATUS = {
  pending: ["Awaiting review", "warn"],
  approved: ["Approved", "ok"],
  rejected: ["Rejected", "bad"],
  blocked: ["Blocked, nothing sent", "bad"],
  failed: ["Model call failed", "bad"],
};

function layerChip(layer) {
  const names = { format: "1 Format", lexicon: "2 Lexicon", combination: "3 Combination", memory: "4 Memory (experimental)" };
  let text = names[layer.name] || layer.name;
  if (layer.score !== null && layer.score !== undefined) text += ` ${layer.score}`;
  if (layer.flags.length) text += `: ${layer.flags.join(", ")}`;
  return chip(text, layer.passed ? "ok" : layer.name === "memory" ? "warn" : "bad");
}

function drawDraft(draft) {
  const [label, kind] = STATUS[draft.status] || [draft.status, ""];
  const box = el("article", { class: "draft" });
  box.append(
    el("h3", {}, [
      el("span", { text: `Draft ${draft.id} · ${draft.case_ref}` }),
      chip(label, kind),
      el("span", { class: "dim", text: `${draft.seconds} s · ${draft.requester}` }),
    ]),
    el("div", { class: "layers" }, draft.gate.layers.map(layerChip)),
  );
  if (draft.gate.generalised.length) {
    box.append(el("p", { class: "label", text: `Generalised before sending: ${draft.gate.generalised.join(", ")} (${draft.gate.rounds} round${draft.gate.rounds === 1 ? "" : "s"})` }));
  }
  box.append(
    el("p", { class: "label", text: "Withheld from the outside model" }),
    el("div", { class: "layers" }, draft.withheld.map((item) =>
      chip(item.last4 ? `${item.kind} ··${item.last4}` : item.kind))),
  );
  if (draft.problem) box.append(el("p", { class: "error", text: draft.problem }));
  if (draft.sent_text) {
    box.append(el("p", { class: "label", text: "Sent to the outside model" }), withCodes(draft.sent_text));
  }
  if (draft.answer) {
    box.append(el("p", { class: "label", text: "Draft, with real values put back here" }), el("pre", { text: draft.answer }));
    if (draft.invented.length || draft.unknown_codes) {
      const parts = [];
      if (draft.invented.length) parts.push(`identifiers the model wrote itself: ${draft.invented.join(", ")}`);
      if (draft.unknown_codes) parts.push(`${draft.unknown_codes} code(s) this request never issued`);
      box.append(el("p", { class: "error", text: `Check before approving. The answer has ${parts.join("; ")}.` }));
    }
  }
  if (draft.status === "pending") {
    const approve = el("button", { class: "primary", type: "button", text: "Approve" });
    const reject = el("button", { type: "button", text: "Reject" });
    approve.onclick = () => review(draft.id, "approve");
    reject.onclick = () => review(draft.id, "reject");
    box.append(el("div", { class: "row" }, [approve, reject]));
  } else if (draft.reviewer) {
    box.append(el("p", { class: "label", text: `Decided by ${draft.reviewer}` }));
  }
  return box;
}

function drawCase(item) {
  const counts = {};
  for (const kind of item.protected) counts[kind] = (counts[kind] || 0) + 1;
  const summary = Object.entries(counts).map(([kind, n]) => `${kind} ×${n}`).join(", ") || "none";
  const value = el("input", { placeholder: "Exact text to forget", "aria-label": `Text to forget from ${item.ref}` });
  const forget = el("button", { type: "button", text: "Forget" });
  const erase = el("button", { type: "button", class: "danger", text: "Erase case" });
  const result = el("p", { class: "label" });
  forget.onclick = async () => {
    try {
      const { forget: done } = await post("/api/forget", { case_ref: item.ref, value: value.value });
      value.value = "";
      showError("");
      result.textContent = `Forgot ${done.copies} cop${done.copies === 1 ? "y" : "ies"} of a ${done.kind}${done.last4 ? ` ending ${done.last4}` : ""} in ${done.ms} ms. Residual against a clean rebuild: ${done.residual}.`;
      await refresh(false);
    } catch (error) { showError(error.message); }
  };
  erase.onclick = async () => {
    try {
      await post("/api/erase", { case_ref: item.ref });
      showError("");
      await refresh();
    } catch (error) { showError(error.message); }
  };
  return el("div", { class: "case" }, [
    el("strong", { text: item.ref }),
    el("span", { class: "label", text: `${item.bytes} bytes in memory · protected: ${summary}` }),
    el("div", { class: "row" }, [value, forget, erase]),
    result,
  ]);
}

function details(entry) {
  const skip = new Set(["seq", "time", "event", "case_ref", "hash", "prev", "sent_text"]);
  return Object.entries(entry)
    .filter(([key, value]) => !skip.has(key) && value !== null && value !== undefined)
    .map(([key, value]) => `${key}=${typeof value === "object" ? JSON.stringify(value) : value}`)
    .join("  ");
}

async function refresh(redrawCases = true) {
  const state = await (await fetch("/api/state")).json();
  $("connector").replaceChildren(chip(`Connector: ${state.connector}`, state.connector === "offline-stub" ? "ok" : "warn"));
  $("memory-mode").replaceChildren(chip(state.block_on_memory ? "Memory check blocks" : "Memory check reports only"));
  $("audit-valid").replaceChildren(chip(state.audit_valid ? "Audit chain intact" : "Audit chain broken", state.audit_valid ? "ok" : "bad"));
  if (redrawCases || !state.cases.length) {
    $("cases").replaceChildren(...(state.cases.length ? state.cases.map(drawCase) : [el("p", { class: "dim", text: "No open cases." })]));
  }
  $("drafts").replaceChildren(...(state.drafts.length ? state.drafts.map(drawDraft) : [el("p", { class: "dim", text: "No drafts yet." })]));
  $("audit").querySelector("tbody").replaceChildren(...[...state.audit].reverse().map((entry) =>
    el("tr", {}, [
      el("td", { text: entry.seq }),
      el("td", { text: entry.time.replace("T", " ").replace("+00:00", " UTC") }),
      el("td", { text: entry.event }),
      el("td", { text: entry.case_ref || "" }),
      el("td", { class: "details", text: details(entry) }),
    ])));
}

async function review(id, decision) {
  try {
    await post("/api/review", { draft: id, decision, reviewer: $("reviewer").value });
    showError("");
    await refresh(false);
  } catch (error) { showError(error.message); }
}

for (const button of document.querySelectorAll("[data-sample]")) {
  button.onclick = async () => {
    const sample = await (await fetch(`/api/sample/${button.dataset.sample}`)).json();
    $("case-text").value = sample.text;
    $("task").value = sample.task;
  };
}

$("ask").onsubmit = async (event) => {
  event.preventDefault();
  const send = $("send");
  send.disabled = true;
  try {
    await post("/api/submit", {
      case_ref: $("case-ref").value,
      requester: $("requester").value,
      task: $("task").value,
      case_text: $("case-text").value,
    });
    $("case-text").value = "";
    showError("");
    await refresh();
  } catch (error) {
    showError(error.message);
  } finally {
    send.disabled = false;
  }
};

refresh().catch(() => showError("Could not reach the gateway."));
