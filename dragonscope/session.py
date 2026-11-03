"""One booth session: guided chat, flagged values and Forget, with no web code.

Chat is guided because the model only understands the wordings it was trained
on. Findings and turns hold byte positions, never values, so ``SessionMemory`` is
the only place a secret lives and Forget has one place to erase it from.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import torch

from dragonscope.activity import ActivityRecorder
from dragonscope.data import (
    FIELDS,
    QUESTIONS,
    REFUSAL,
    STATEMENTS,
    split_names,
)
from dragonscope.detect import digits_of, find_in_reply, find_stated
from dragonscope.engine import SessionMemory
from dragonscope.evaluate_forget import MAX_REPLY_BYTES
from dragonscope.vendor.bdh.bdh import BDH

# Replies get less reliable as a session grows. On the trained model the first
# wrong reply came at byte 406. 93% were correct at bytes 480-519, and 61% at 600-639.
SESSION_LIMIT = 400
# The longest trained reply and its newline: a card with separators.
REPLY_RESERVE = 41
NEWLINE = 10
# Time each byte is held when paced. Computing one, comparison included, took about
# 45 to 85 ms on the demo laptop on battery with the page open.
READ_PACE = 0.075
WRITE_PACE = 0.11


class SessionError(Exception):
    """A request the session refused. Messages never repeat the visitor's value."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    start: int
    end: int


@dataclass
class Finding:
    id: int
    kind: str
    source: str  # "user" or "reply"
    turn: int
    start: int
    end: int
    luhn_valid: bool | None


def guide() -> dict:
    """What the guided chat can offer. Everything here is a trained wording."""
    first_names, last_names, branches = split_names("train")
    return {
        "fields": list(FIELDS),
        "statements": {field: STATEMENTS[field][0] for field in FIELDS},
        "questions": {field: QUESTIONS[field][0] for field in FIELDS},
        "first_names": first_names,
        "last_names": last_names,
        "branches": branches,
        "refusal": REFUSAL,
    }


def _check_value(field: str, value: str) -> str:
    """Return the value in trained form, or refuse it before the model reads it."""
    value = value.strip()
    if field in ("card", "nic"):
        value = value.upper()
        found = find_stated(value)
        if not (
            len(found) == 1
            and found[0].kind == field
            and found[0].start == 0
            and found[0].end == len(value)
        ):
            label = "card number" if field == "card" else "NIC"
            raise SessionError("invalid_value", f"That is not a valid {label}.")
        return value
    first_names, last_names, branches = split_names("train")
    if field == "name":
        parts = value.split(" ")
        valid = len(parts) == 2 and parts[0] in first_names and parts[1] in last_names
    else:
        valid = value in branches
    if not valid:
        raise SessionError("invalid_value", f"Please choose a listed {field}.")
    return value


class DemoSession:
    def __init__(self, model: BDH):
        self.memory = SessionMemory(model)
        self.emit = None  # Called with each frame as it is computed. Keeps no copy.
        self.paced = False  # Hold each byte long enough to watch it.
        self.full_speed = False  # The visitor's choice to drop that pacing.
        self.reset()

    def reset(self) -> None:
        self.memory.reset()
        self.turns: list[Turn] = []
        self.findings: list[Finding] = []
        self.deletion_log: list[dict] = []
        self.stated: set[str] = set()
        self.last_answer: dict | None = None
        self._next_id = 1

    # Chat

    def say(self, field: str, value: str) -> None:
        """State a field using its trained wording; the model then replies.

        A field can be stated once per session, even after Forget: recall of a
        second value was exact in only 13 of 20 tries on the trained model.
        """
        self._check_field(field)
        if field in self.stated:
            raise SessionError(
                "already_stated", f"The {field} was already given. Reset to enter another."
            )
        value = _check_value(field, value)
        self._exchange(STATEMENTS[field][0].format(value), question=False)
        self.stated.add(field)
        self.last_answer = None

    def ask(self, field: str) -> None:
        """Ask for a field using its trained wording."""
        self._check_field(field)
        reply = self._exchange(QUESTIONS[field][0], question=True)
        self.last_answer = {"field": field, "refused": reply == REFUSAL}

    @staticmethod
    def _check_field(field: str) -> None:
        if field not in FIELDS:
            raise SessionError("unknown_field", "Unknown field.")

    def _exchange(self, line: str, question: bool) -> str:
        memory = self.memory
        prompt = f"U: {line}\nA: ".encode("ascii")
        if memory.position + len(prompt) + REPLY_RESERVE > SESSION_LIMIT:
            raise SessionError("session_full", "This session is full. Please reset.")

        recorder = ActivityRecorder(memory.model, memory.n_layer)
        shadow = self._counterfactual_memory()
        memory.observer = recorder
        if shadow is not None:
            shadow.observer = recorder.watch_counterfactual
        try:
            start = memory.position + len(b"U: ")
            end = start + len(line)
            for position, byte in enumerate(prompt, start=memory.position):
                shown = chr(byte) if start <= position < end else ""
                self._compute_byte(byte, "read", shown, question, recorder, shadow)
            self.turns.append(Turn("user", start, end))
            for detection in find_stated(line):
                self._add_finding(
                    detection.kind,
                    "user",
                    start + detection.start,
                    start + detection.end,
                    detection.luhn_valid,
                )

            start = memory.position
            written = bytearray()
            finished = False
            for _ in range(MAX_REPLY_BYTES):
                byte = self._choose_byte()
                finished = byte == NEWLINE
                shown = "" if finished else chr(byte) if byte < 128 else "\ufffd"
                self._compute_byte(byte, "write", shown, question, recorder, shadow)
                if finished:
                    break
                written.append(byte)
            if not finished:
                self._compute_byte(NEWLINE, "write", "", question, recorder, shadow)
        finally:
            memory.observer = None

        self.turns.append(Turn("assistant", start, start + len(written)))
        # latin-1 maps each byte to one character, so offsets stay byte positions.
        reply = written.decode("latin-1")
        detection = find_in_reply(reply)
        if detection is not None:
            self._add_finding(
                detection.kind,
                "reply",
                start + detection.start,
                start + detection.end,
                detection.luhn_valid,
            )
        return reply

    def _choose_byte(self) -> int:
        """Greedy output, as in ``SessionMemory.generate``."""
        return int(self.memory.next_logits.argmax())

    def _counterfactual_memory(self) -> SessionMemory | None:
        """Return a copy of memory that never read the flagged bytes, if any exist."""
        positions = self._flagged_positions()
        if not positions:
            return None
        shadow = self.memory.fork()
        for position in positions:
            shadow.forget(position, position + 1, "subtract")
        shadow.rebuild()
        return shadow

    def _compute_byte(
        self,
        byte: int,
        phase: str,
        shown: str,
        question: bool,
        recorder: ActivityRecorder,
        shadow: SessionMemory | None,
    ) -> None:
        """Compute one byte for real, then report it at once, paced if asked."""
        memory = self.memory
        chosen_from = memory.next_logits
        without_from = None if shadow is None else shadow.next_logits
        started = time.perf_counter()
        if shadow is not None:
            shadow.read(bytes([byte]))
        memory.read(bytes([byte]))
        computed = time.perf_counter() - started

        depends = 0.0
        if phase == "write" and without_from is not None:
            p_with = chosen_from.softmax(dim=0)[byte]
            p_without = without_from.softmax(dim=0)[byte]
            depends = round(max(0.0, float(p_with - p_without)), 3)
        pace = 0.0
        if self.paced and not self.full_speed:
            pace = READ_PACE if phase == "read" else WRITE_PACE
        frame = recorder.take_frame()
        frame.update(
            type="frame",
            phase=phase,
            char=shown,
            question=question,
            depends=depends,
            ms=round(computed * 1000, 1),
            show_ms=round(max(computed, pace) * 1000),
        )
        if self.emit is not None:
            self.emit(frame)
        if pace > computed:
            time.sleep(pace - computed)

    def _flagged_positions(self) -> set[int]:
        return {
            position
            for finding in self.findings
            for position in range(finding.start, finding.end)
        }

    def _flagged_rho(self) -> torch.Tensor | None:
        """Return the part of ``rho`` written by flagged bytes, if there are any."""
        memory = self.memory
        positions = self._flagged_positions()
        if not positions:
            return None
        flagged = torch.zeros_like(memory.rho)
        for position in positions:
            entry = memory.ledger[position]
            for level in range(memory.n_layer):
                flagged[level] += torch.einsum(
                    "d,hn->hdn", entry.v[level], entry.qr[level]
                )
        return flagged

    def _add_finding(
        self, kind: str, source: str, start: int, end: int, luhn: bool | None
    ) -> None:
        self.findings.append(
            Finding(self._next_id, kind, source, len(self.turns) - 1, start, end, luhn)
        )
        self._next_id += 1

    def _value(self, finding: Finding) -> str:
        return "".join(
            chr(self.memory.bytes[position])
            for position in range(finding.start, finding.end)
        )

    # Forget

    def _linked_findings(self, target: Finding) -> list[Finding]:
        """Return every finding that one Forget must remove.

        That means each copy of the value in any format, and each number the
        model wrote that is not exactly a different value the visitor still has
        on record. A garbled echo cannot be traced to its source, so it goes too.
        """
        wanted = digits_of(self._value(target))
        keep = {
            digits_of(self._value(f))
            for f in self.findings
            if f.source == "user" and digits_of(self._value(f)) != wanted
        }
        return [
            f
            for f in self.findings
            if digits_of(self._value(f)) == wanted
            or (f.source == "reply" and digits_of(self._value(f)) not in keep)
        ]

    def forget(self, finding_id: int) -> dict:
        """Run ``subtract``, then rebuild from the surviving bytes, and report it."""
        target = next((f for f in self.findings if f.id == finding_id), None)
        if target is None:
            raise SessionError("unknown_finding", "That item is no longer on record.")

        memory = self.memory
        secret = digits_of(self._value(target))
        kind = target.kind
        linked = self._linked_findings(target)
        before = memory.rho.clone()

        started = time.perf_counter()
        erased = 0
        for finding in linked:
            erased += len(memory.forget(finding.start, finding.end, "subtract"))
        subtract_ms = (time.perf_counter() - started) * 1000

        # Rebuild a copy first, so one rebuild also gives the residual left by
        # subtract. The copy has nothing to do with the old ledger or logits.
        started = time.perf_counter()
        clean, residual_after_subtract = self._distance_from_rebuild()
        memory.rho = clean.rho
        memory.ledger = clean.ledger
        rebuild_ms = (time.perf_counter() - started) * 1000

        linked_ids = {finding.id for finding in linked}
        self.findings = [f for f in self.findings if f.id not in linked_ids]
        self.last_answer = None
        entry = {
            "type": kind,
            "last4": secret[-4:],
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.deletion_log.append(entry)

        # A second rebuild, so this is a measurement and not an assumption.
        _, residual_after_rebuild = self._distance_from_rebuild()
        erased_positions = {
            position
            for finding in linked
            for position in range(finding.start, finding.end)
        }
        change = float(
            (before - memory.rho).norm() / before.norm().clamp_min(1e-12)
        )
        return {
            "log_entry": dict(entry),
            "copies_erased": len(linked),
            "bytes_erased": erased,
            "memory_change": change,
            "residual_after_subtract": residual_after_subtract,
            "residual_after_rebuild": residual_after_rebuild,
            "subtract_ms": round(subtract_ms, 1),
            "rebuild_ms": round(rebuild_ms, 1),
            "checks": {
                "value_in_memory_bytes": secret in digits_of(
                    memory.text().decode("latin-1")
                ),
                "ledger_entries_left": len(erased_positions & memory.ledger.keys()),
                "pending_logits": memory.next_logits is not None,
            },
        }

    def _distance_from_rebuild(self) -> tuple[SessionMemory, float]:
        memory = self.memory
        clean = memory.fork()
        clean.rebuild()
        distance = (memory.rho - clean.rho).norm() / clean.rho.norm().clamp_min(1e-12)
        return clean, float(distance)

    # What the dashboard sees

    def _segments(self, turn_index: int, turn: Turn) -> list[dict]:
        """Split a turn into plain, flagged and forgotten runs of text."""
        owner = {
            position: finding.id
            for finding in self.findings
            if finding.turn == turn_index
            for position in range(finding.start, finding.end)
        }
        segments: list[dict] = []
        previous = None
        for position in range(turn.start, turn.end):
            byte = self.memory.bytes.get(position)
            if byte is None:
                label = {"forgotten": True}
            elif position in owner:
                label = {"finding": owner[position]}
            else:
                label = {}
            if label != previous:
                segments.append({"text": "", **label})
                previous = label
            if byte is not None:
                segments[-1]["text"] += chr(byte) if byte < 128 else "�"
        return segments

    def _neuron_memory(self) -> dict[str, list[int]]:
        """Return, per neuron from 0 to 100, what it holds and how much is flagged.

        A neuron has one memory per round, and these are combined here.
        """
        memory = self.memory
        strength = memory.rho.norm(dim=(0, 2)).flatten()
        flagged = self._flagged_rho()
        share = (
            torch.zeros_like(strength)
            if flagged is None
            else flagged.norm(dim=(0, 2)).flatten() / strength.clamp_min(1e-12)
        )
        scaled = strength / strength.max().clamp_min(1e-12)
        return {
            "strength": (scaled * 100).round().int().tolist(),
            "flagged": (share.clamp_max(1.0) * 100).round().int().tolist(),
        }

    def state(self) -> dict:
        """Return the whole visible state. Clients replace theirs with this."""
        memory = self.memory
        return {
            "type": "state",
            "turns": [
                {"role": turn.role, "segments": self._segments(index, turn)}
                for index, turn in enumerate(self.turns)
            ],
            "findings": [
                {
                    "id": finding.id,
                    "kind": finding.kind,
                    "source": finding.source,
                    "turn": finding.turn,
                    "luhn_valid": finding.luhn_valid,
                    "last4": digits_of(self._value(finding))[-4:],
                }
                for finding in self.findings
            ],
            "stated": sorted(self.stated),
            "full_speed": self.full_speed,
            "last_answer": self.last_answer,
            "deletion_log": [dict(entry) for entry in self.deletion_log],
            "memory": {
                "bytes_stored": len(memory.bytes),
                "ledger_entries": len(memory.ledger),
                "position": memory.position,
                "position_limit": SESSION_LIMIT,
                "rho_norm": round(float(memory.rho.norm()), 4),
                **self._neuron_memory(),
            },
        }

