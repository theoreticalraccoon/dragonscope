"""See, audit and edit the memory of a BDH model.

    import dragonscope as ds

    memory = ds.capture(model)
    memory.read("My card number is 4111 1111 1111 1111.\\n")
    edit = memory.forget("4111 1111 1111 1111")
    print(edit.seconds, edit.residual)

``capture`` works with the BDH model from pathwaycom/bdh, of any size. Other
implementations have not been tested and may need an adapter.

Memory: ``subtract`` needs a ledger entry for each byte it may erase, and that
grows with the session (about 68 KB a byte for the demo model, 0.8 MB for the
published BDH size). ``ledger="compact"`` cuts this at least 2 times (2.7 for the demo model), and
``read(text, ledger=False)`` skips text that will never need forgetting.
Forgetting unrecorded text falls back to ``rebuild``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from dragonscope.engine import LEDGER_MODES, SessionMemory
from dragonscope.vendor.bdh.bdh import BDH

METHODS = ("auto", "subtract", "rebuild")
Target = str | bytes | tuple[int, int]


def _as_bytes(data: str | bytes) -> bytes:
    return data.encode("utf-8") if isinstance(data, str) else data


@dataclass
class Step:
    """What the model did for one byte. It holds no text once the byte is forgotten."""

    position: int
    byte: int | None
    firing: list[int]  # Neurons firing, per round.
    passing: list[int]  # Neurons passing a signal on, per round.
    rho_norm: float
    state: torch.Tensor | None = None  # A copy of rho, only with keep_states=True.


@dataclass
class ForgetResult:
    method: str  # The method used: "subtract" or "rebuild".
    copies: int
    bytes_erased: int
    seconds: float
    residual: float | None  # Relative distance from never having read the bytes.
    residual_per_round: list[float] = field(default_factory=list)


class MemoryCapture:
    """A BDH session whose memory can be recorded, inspected and edited.

    ``keep_states`` copies the whole memory at every byte, for later analysis. That
    is 8 MB a byte for the DragonScope model, and far more for larger ones.
    ``ledger`` is ``"exact"``, ``"compact"`` or ``"off"``; see the module docstring.
    """

    def __init__(self, model: BDH, keep_states: bool = False, ledger: str = "exact"):
        if ledger not in LEDGER_MODES:
            raise ValueError(f"unknown ledger mode {ledger!r}: use one of {LEDGER_MODES}")
        self.memory = SessionMemory(model, ledger)
        self.keep_states = keep_states
        self.history: list[Step] = []
        self._firing: list[int] = []
        self._passing: list[int] = []
        self._observers: list = []
        self.memory.observer = self._observe

    @property
    def model(self) -> BDH:
        return self.memory.model

    @property
    def rho(self) -> torch.Tensor:
        """The memory: a compressed form of the synaptic state, per round and head."""
        return self.memory.rho

    @property
    def position(self) -> int:
        return self.memory.position

    @property
    def ledger_bytes(self) -> int:
        """Memory held by the ledger that ``subtract`` uses, in bytes."""
        return self.memory.ledger_bytes()

    @property
    def text(self) -> str:
        """The surviving text. Forgotten bytes are simply absent."""
        return self.memory.text().decode("utf-8", errors="replace")

    def add_observer(self, observer) -> None:
        """Also call ``observer(round, x_sparse, xy_sparse, qr)`` during each byte."""
        self._observers.append(observer)

    def _observe(self, level: int, x_sparse, xy_sparse, qr) -> None:
        self._firing.append(int((x_sparse > 0).sum()))
        self._passing.append(int((xy_sparse > 0).sum()))
        for observer in self._observers:
            observer(level, x_sparse, xy_sparse, qr)

    def _record(self, data: bytes) -> None:
        memory = self.memory
        for byte in data:
            self._firing, self._passing = [], []
            position = memory.position
            memory.read(bytes([byte]))
            self.history.append(
                Step(
                    position,
                    byte,
                    self._firing,
                    self._passing,
                    float(memory.rho.norm()),
                    memory.rho.clone() if self.keep_states else None,
                )
            )

    def _ledger(self, ledger: bool | None):
        """Switch ledger recording for one call, returning the previous setting."""
        previous = self.memory.recording
        if ledger is not None:
            self.memory.recording = bool(ledger)
        return previous

    def read(self, data: str | bytes, ledger: bool | None = None) -> "MemoryCapture":
        """Read text into memory, recording each byte.

        ``ledger=False`` keeps no ledger entries for this text, to save memory.
        It can still be forgotten, by ``rebuild``.
        """
        data = _as_bytes(data)
        if not data:
            raise ValueError("read() needs at least one byte")
        previous = self._ledger(ledger)
        try:
            self._record(data)
        finally:
            self.memory.recording = previous
        return self

    def generate(
        self, max_bytes: int = 70, stop: str | None = "\n", ledger: bool | None = None
    ) -> str:
        """Write greedily, stopping after ``stop``. What is written enters memory."""
        if self.memory.next_logits is None:
            raise RuntimeError("read something first: forgetting clears the pending output")
        stop_byte = None if stop is None else _as_bytes(stop)[0]
        output = bytearray()
        previous = self._ledger(ledger)
        try:
            for _ in range(max_bytes):
                byte = int(self.memory.next_logits.argmax())
                self._record(bytes([byte]))
                output.append(byte)
                if byte == stop_byte:
                    break
        finally:
            self.memory.recording = previous
        return output.decode("utf-8", errors="replace")

    def next_byte_probabilities(self) -> torch.Tensor:
        """Return the model's probability for each possible next byte."""
        if self.memory.next_logits is None:
            raise RuntimeError("read something first: forgetting clears the pending output")
        return self.memory.next_logits.softmax(dim=0)

    def find(self, value: str | bytes) -> list[tuple[int, int]]:
        """Return the ``(start, end)`` positions of each copy of ``value``."""
        return self.memory.find_positions(_as_bytes(value))

    def forget(
        self, target: Target, method: str = "auto", measure: bool = True
    ) -> ForgetResult:
        """Remove every copy of a value, or one ``(start, end)`` range, from memory.

        ``subtract`` removes the bytes' own traces, at a cost set by how much is
        erased. It leaves a residual in rounds after the first, and needs ledger
        entries. ``rebuild`` recomputes memory from the surviving text, at a cost set
        by how long the session is, and matches a session that never read the bytes.
        ``auto`` uses ``subtract`` when every byte has a ledger entry, else ``rebuild``.
        ``measure`` also computes the residual, which itself costs one replay and is
        not included in ``seconds``.
        """
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}: use one of {METHODS}")
        spans = [target] if isinstance(target, tuple) else self.find(target)
        if not spans:
            raise ValueError("that value is not in memory")

        engine = self.memory
        recorded = all(
            position in engine.ledger
            for start, end in spans
            for position in range(start, end)
            if position in engine.bytes
        )
        if method == "auto":
            method = "subtract" if recorded else "rebuild"
        elif method == "subtract" and not recorded:
            raise ValueError("some of this text has no ledger entries: use rebuild or auto")

        started = time.perf_counter()
        erased: list[int] = []
        if method == "subtract":
            for start, end in spans:
                erased += engine.forget(start, end, "subtract")
        else:
            for start, end in spans:
                erased += engine.forget(start, end, "replay", defer_rebuild=True)
            engine.rebuild()
        seconds = time.perf_counter() - started

        self._scrub_history(erased)
        residual = self.residual() if measure else None
        return ForgetResult(
            method,
            len(spans),
            len(erased),
            seconds,
            None if residual is None else residual["total"],
            [] if residual is None else residual["per_round"],
        )

    def _scrub_history(self, erased: list[int]) -> None:
        """Leave no copy of forgotten data in what was recorded.

        Every state from the first erased byte onward holds its trace, so those go.
        """
        if not erased:
            return
        gone = set(erased)
        first = min(erased)
        self.history = [step for step in self.history if step.position not in gone]
        for step in self.history:
            if step.position >= first:
                step.state = None

    def rebuild(self) -> None:
        """Recompute memory from the surviving text, clearing any residual."""
        self.memory.rebuild()

    def residual(self) -> dict:
        """Measure the distance from a session that never read what was forgotten."""
        measured = self.memory.residual_from_replay()
        return {"total": measured["total"], "per_round": measured["per_layer"]}

    def neuron_strength(self) -> torch.Tensor:
        """Return how much each neuron holds, shaped (head, neuron), across rounds."""
        return self.memory.rho.norm(dim=(0, 2))

    def fork(self) -> "MemoryCapture":
        """Return an independent copy, to try something without changing this session.

        A fork records from this point, and starts with an empty history.
        """
        other = MemoryCapture(
            self.model, keep_states=self.keep_states, ledger=self.memory.ledger_mode
        )
        observer = other.memory.observer
        other.memory = self.memory.fork()
        other.memory.observer = observer
        return other

    def reset(self) -> None:
        """Clear memory and everything recorded."""
        self.memory.reset()
        self.history = []


def capture(model: BDH, keep_states: bool = False, ledger: str = "exact") -> MemoryCapture:
    """Start recording a BDH model's memory."""
    return MemoryCapture(model, keep_states=keep_states, ledger=ledger)
