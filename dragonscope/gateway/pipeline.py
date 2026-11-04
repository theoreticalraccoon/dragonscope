"""The request lifecycle: load, register, replace, gate, send, re-link, review, log.

Identifiable data is handled only inside this process. What reaches the
connector is the cleared text, after the leak gate allowed it. The placeholder
map is destroyed when each request ends, whatever happens.
"""

from __future__ import annotations

import itertools
import re
import threading
import time
from dataclasses import dataclass, field

from dragonscope.gateway.audit import AuditLog
from dragonscope.gateway.connector import SYSTEM_PROMPT, Connector, ConnectorError, OfflineStub
from dragonscope.gateway.gate import LeakGate, ProtectedItem, protected_items
from dragonscope.gateway.identifiers import Identifier, identify
from dragonscope.gateway.placeholders import PlaceholderMap, deidentify
from dragonscope.toolkit import MemoryCapture
from dragonscope.vendor.bdh.bdh import BDH

MAX_CASE_BYTES = 2000
MAX_TASK_BYTES = 500
CASE_REF = re.compile(r"[A-Za-z0-9_-]{1,32}")
FORGOTTEN = "[forgotten]"


class GatewayError(Exception):
    """A request the gateway refused. Messages never repeat what was submitted."""


@dataclass
class Case:
    ref: str
    memory: MemoryCapture  # The case's synaptic memory, and its only copy of the text.
    items: list[ProtectedItem]
    values: list[str]  # Direct identifiers, for the gate's known-value check.

    @property
    def text(self) -> str:
        return self.memory.text

    def pairs(self) -> list[tuple[int, int]]:
        return sorted(self.memory.memory.bytes.items())


@dataclass
class Draft:
    id: int
    case_ref: str
    requester: str
    task: str
    status: str  # "blocked", "pending", "approved", "rejected" or "failed".
    withheld: list[dict]
    gate: dict
    sent_text: str | None = None
    answer: str | None = None
    invented: list[str] = field(default_factory=list)
    unknown_codes: int = 0
    reviewer: str | None = None
    seconds: float = 0.0
    problem: str | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _withheld(found: list[Identifier], generalised: list[str]) -> list[dict]:
    items = [
        {"kind": item.kind, "last4": item.last4}
        for item in found
        if item.kind not in ("age", "occupation", "condition")
    ]
    items += [{"kind": kind, "last4": None} for kind in generalised]
    return items


class Gateway:
    """One institution's gateway: cases, drafts awaiting review, and the audit log."""

    def __init__(
        self,
        model: BDH,
        connector: Connector | None = None,
        gate: LeakGate | None = None,
        audit: AuditLog | None = None,
    ):
        self.model = model
        self.connector = connector or OfflineStub()
        self.gate = gate or LeakGate(model)
        self.audit = audit or AuditLog()
        self.cases: dict[str, Case] = {}
        self.drafts: dict[int, Draft] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    # Cases

    def open_case(self, ref: str, text: str) -> Case:
        """Read a case into its own memory. Opening a ref again replaces it."""
        if not isinstance(ref, str) or not CASE_REF.fullmatch(ref) or identify(ref):
            raise GatewayError("Use a case reference of letters, digits, - or _, with no identifiers.")
        data = text.encode("utf-8") if isinstance(text, str) else b""
        if not data.strip():
            raise GatewayError("The case is empty.")
        if len(data) > MAX_CASE_BYTES:
            raise GatewayError(f"Cases are limited to {MAX_CASE_BYTES} bytes in this prototype.")
        with self._lock:
            if ref in self.cases:
                self._erase(ref, log=False)
            memory = MemoryCapture(self.model, ledger="off")
            engine = memory.memory
            # One batched pass, the same one rebuild uses, instead of byte by byte.
            engine.bytes = dict(enumerate(data))
            engine.position = len(data)
            engine.rebuild()
            items, values = protected_items(text)
            case = Case(ref, memory, items, values)
            self.cases[ref] = case
            self.audit.append(
                "open", case_ref=ref, bytes=len(data),
                protected=sorted({item.kind for item in items}),
            )
            return case

    def _case(self, ref: str) -> Case:
        case = self.cases.get(ref)
        if case is None:
            raise GatewayError("That case is not open.")
        return case

    # Requests

    def submit(self, case_ref: str, task: str, requester: str) -> Draft:
        if not isinstance(task, str) or not task.strip():
            raise GatewayError("Say what the draft is for.")
        if len(task.encode("utf-8")) > MAX_TASK_BYTES:
            raise GatewayError(f"Tasks are limited to {MAX_TASK_BYTES} bytes.")
        if not isinstance(requester, str) or not requester.strip():
            raise GatewayError("Sign in with your name and role.")
        with self._lock:
            case = self._case(case_ref)
            started = time.perf_counter()
            placeholders = PlaceholderMap()
            try:
                return self._submit(case, task, requester.strip(), placeholders, started)
            finally:
                placeholders.destroy()

    def _submit(
        self, case: Case, task: str, requester: str, placeholders: PlaceholderMap, started: float
    ) -> Draft:
        cleared_task, found_task = deidentify(task, placeholders)
        cleared_case, found_case = deidentify(case.text, placeholders)
        prompt = f"Task: {cleared_task}\n\nCase:\n{cleared_case}"
        result = self.gate.clear(
            prompt, case.values, case.pairs(), case.items, system=SYSTEM_PROMPT
        )
        draft = Draft(
            id=next(self._ids),
            case_ref=case.ref,
            requester=requester,
            task=task,
            status="blocked",
            withheld=_withheld(found_task + found_case, result.generalised),
            gate=result.summary(),
        )
        if result.allowed:
            draft.sent_text = result.text
            try:
                answer = self.connector.complete(SYSTEM_PROMPT, result.text)
            except ConnectorError as error:
                draft.status = "failed"
                draft.problem = str(error)
            else:
                relinked = placeholders.relink(answer)
                draft.answer = relinked.text
                draft.invented = relinked.invented
                draft.unknown_codes = relinked.unknown_codes
                draft.status = "pending"
        else:
            draft.problem = "The leak gate blocked this request, so nothing was sent."
        draft.seconds = round(time.perf_counter() - started, 3)
        self.drafts[draft.id] = draft
        self.audit.append(
            "request",
            request=draft.id,
            case_ref=case.ref,
            requester=requester,
            status=draft.status,
            withheld=draft.withheld,
            gate={layer["name"]: layer["passed"] for layer in draft.gate["layers"]},
            memory_score=draft.gate["layers"][3]["score"],
            rounds=result.rounds,
            connector=self.connector.name if result.allowed else None,
            sent_bytes=len(draft.sent_text.encode("utf-8")) if draft.sent_text else 0,
            sent_text=draft.sent_text,
        )
        return draft

    def decide(self, draft_id: int, reviewer: str, approve: bool) -> Draft:
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise GatewayError("A named reviewer must decide.")
        with self._lock:
            draft = self.drafts.get(draft_id)
            if draft is None:
                raise GatewayError("That draft does not exist.")
            if draft.status != "pending":
                raise GatewayError("Only a pending draft can be approved or rejected.")
            draft.status = "approved" if approve else "rejected"
            draft.reviewer = reviewer.strip()
            self.audit.append(
                "review", request=draft.id, case_ref=draft.case_ref,
                reviewer=draft.reviewer, decision=draft.status,
            )
            return draft

    # Forgetting and erasure

    def forget_fact(self, case_ref: str, value: str) -> dict:
        """Remove one value from a case's memory and drafts, measuring what is left."""
        if not isinstance(value, str) or not value.strip():
            raise GatewayError("Choose the text to forget.")
        with self._lock:
            case = self._case(case_ref)
            if not case.memory.find(value):
                raise GatewayError("That text is not in this case.")
            found = identify(value)
            kind = found[0].kind if found else "text"
            last4 = found[0].last4 if found else None
            result = case.memory.forget(value, method="rebuild")
            # Items wholly inside forgotten bytes are gone; the rest keep their spans.
            alive = set(case.memory.memory.bytes)
            case.items = [
                item for item in case.items
                if any(p in alive for p in range(item.start, item.end))
            ]
            case.values = [v for v in case.values if v != value and v in case.text]
            for draft in self.drafts.values():
                if draft.case_ref == case_ref:
                    draft.task = draft.task.replace(value, FORGOTTEN)
                    if draft.answer:
                        draft.answer = draft.answer.replace(value, FORGOTTEN)
            self.audit.append(
                "forget", case_ref=case_ref, kind=kind, last4=last4,
                copies=result.copies, residual=result.residual,
            )
            return {
                "kind": kind,
                "last4": last4,
                "copies": result.copies,
                "bytes_erased": result.bytes_erased,
                "residual": result.residual,
                "ms": round(result.seconds * 1000, 1),
            }

    def erase_case(self, case_ref: str) -> None:
        """Erase a person: the case memory, its drafts and the text sent for it."""
        with self._lock:
            self._case(case_ref)
            self._erase(case_ref, log=True)

    def _erase(self, case_ref: str, log: bool) -> None:
        case = self.cases.pop(case_ref)
        case.memory.reset()
        case.items.clear()
        case.values.clear()
        for draft_id in [d.id for d in self.drafts.values() if d.case_ref == case_ref]:
            del self.drafts[draft_id]
        self.audit.blank_case(case_ref)
        if log:
            self.audit.append("erase", case_ref=case_ref)

    # What the console sees

    def state(self) -> dict:
        with self._lock:
            return {
                "cases": [
                    {
                        "ref": case.ref,
                        "bytes": len(case.memory.memory.bytes),
                        "protected": [item.kind for item in case.items],
                        "rho_norm": round(float(case.memory.rho.norm()), 2),
                    }
                    for case in self.cases.values()
                ],
                "drafts": [draft.to_dict() for draft in reversed(self.drafts.values())],
                "audit": self.audit.entries[-50:],
                "audit_valid": self.audit.verify(),
                "connector": self.connector.name,
                "block_on_memory": self.gate.block_on_memory,
            }
