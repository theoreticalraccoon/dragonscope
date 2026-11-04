"""The leak gate: the last check before any text leaves.

Four layers check the outgoing text, and any flag blocks the send:

1. **Format rules**, broader than the de-identifier's: any card-, NIC- or
   phone-shaped text, any email, any run of 5 or more digits, any full date,
   any age over 89.
2. **Lexicon and known values**: listed names and places, names after a title or
   label, and any protected value from the case appearing again.
3. **Combination risk**: quasi-identifiers are scored. Above the threshold the
   text is generalised and rechecked, at most twice, and then a person decides.
4. **Memory activation (experimental)**: BDH reads the case and then the
   outgoing text. For each protected item, the score is how much attention the
   outgoing text pays to that item's positions in memory, relative to its share
   of the case. This equals the item's share of ``rho @ QR_t`` in the engine's
   terms. Whether it catches paraphrases that layers 1 to 3 miss is unproven, so
   it only reports unless ``block_on_memory`` is set.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import torch

from dragonscope.detect import CARD, NIC
from dragonscope.engine import SessionMemory
from dragonscope.gateway.identifiers import (
    DATES,
    DIRECT,
    EMAIL,
    LABEL_NAME,
    LEXICON_NAME,
    PHONE,
    PLACE,
    TITLE_NAME,
    generalise_quasi,
    identify,
)
from dragonscope.vendor.bdh.bdh import BDH

MAX_ROUNDS = 2
DIGIT_RUN = re.compile(r"\d(?:[\s-]?\d){4,}")
# "90+" is the generalised form, so it is allowed.
OLD_AGE = re.compile(
    r"\b(?:9\d|1[0-4]\d)(?=[- ]years?[- ]old\b|[- ]year-old\b)"
    r"|\b[Aa]ged?\s*:?\s*(?:9\d|1[0-4]\d)\b(?!\+)"
)
WEIGHTS = {"occupation": 2.0, "condition": 2.0, "age": 1.0, "year": 0.0}
RISK_THRESHOLD = 2.0
# A protected item drawing this many times its share of attention is flagged.
MEMORY_THRESHOLD = 3.0
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass
class ProtectedItem:
    """A finding in the case, by kind and byte span. The value stays in the case."""

    kind: str
    start: int
    end: int


@dataclass
class Layer:
    name: str
    passed: bool
    flags: list[str] = field(default_factory=list)  # Kinds only, never values.
    score: float | None = None


@dataclass
class GateResult:
    allowed: bool
    text: str
    rounds: int
    needs_review: bool
    layers: list[Layer]
    generalised: list[str] = field(default_factory=list)  # Kinds generalised.

    def summary(self) -> dict:
        return {
            "allowed": self.allowed,
            "rounds": self.rounds,
            "needs_review": self.needs_review,
            "layers": [asdict(layer) for layer in self.layers],
            "generalised": self.generalised,
        }


def format_layer(text: str) -> Layer:
    flags = []
    checks = (
        ("card", CARD),
        ("nic", NIC),
        ("phone", PHONE),
        ("email", EMAIL),
        ("number", DIGIT_RUN),
        ("old_age", OLD_AGE),
    )
    for kind, pattern in checks:
        if pattern.search(text):
            flags.append(kind)
    # The de-identifier leaves only years, which no date pattern matches.
    if any(pattern.search(text) for pattern in DATES):
        flags.append("date")
    return Layer("format", not flags, flags)


def lexicon_layer(text: str, protected_values: list[str]) -> Layer:
    flags = []
    if LEXICON_NAME.search(text) or TITLE_NAME.search(text) or LABEL_NAME.search(text):
        flags.append("name")
    if PLACE.search(text):
        flags.append("place")
    lowered = text.lower()
    if any(len(value) >= 3 and value.lower() in lowered for value in protected_values):
        flags.append("protected_value")
    return Layer("lexicon", not flags, flags)


def risk_score(text: str, weights: dict[str, float]) -> tuple[float, list[str]]:
    kinds = [f.kind for f in identify(text) if f.kind in weights]
    kinds += ["year"] * len(YEAR.findall(text))
    return sum(weights.get(kind, 0.0) for kind in kinds), sorted(set(kinds))


@torch.no_grad()
def memory_scores(
    model: BDH,
    case: list[tuple[int, int]],
    items: list[ProtectedItem],
    outgoing: bytes,
) -> list[float]:
    """Score each protected item by the attention the outgoing text pays it.

    ``case`` is the surviving ``(position, byte)`` pairs of the case memory. The
    outgoing text is placed after the case, so it reads with the case in memory.
    A score of 1 means the item gets its fair share of attention, and higher
    means the outgoing text keeps reaching for it.
    """
    if not case or not outgoing or not items:
        return [0.0 for _ in items]
    engine = SessionMemory(model, "off")
    after = case[-1][0] + 1
    pairs = case + [(after + offset, byte) for offset, byte in enumerate(outgoing)]
    _, _, _, qr = engine._forward_all(pairs)  # n_layer, n_head, T, N
    count = len(case)
    slot = {position: index for index, (position, _) in enumerate(case)}
    masks = []
    for item in items:
        mask = torch.zeros(count, dtype=qr.dtype, device=qr.device)
        for position in range(item.start, item.end):
            if position in slot:
                mask[slot[position]] = 1.0
        masks.append(mask)
    share = torch.zeros(len(items), len(outgoing), dtype=qr.dtype, device=qr.device)
    for level in range(qr.shape[0]):
        weights = (qr[level, :, count:] @ qr[level, :, :count].mT).abs()  # H, M, n
        total = weights.sum(dim=-1).clamp_min(1e-12)  # H, M
        for index, mask in enumerate(masks):
            share[index] += ((weights @ mask) / total).mean(dim=0)
    share /= qr.shape[0]
    scores = []
    for index, mask in enumerate(masks):
        fair = float(mask.sum()) / count
        if fair == 0.0:
            scores.append(0.0)
            continue
        scores.append(float(torch.quantile(share[index], 0.95)) / fair)
    return scores


class LeakGate:
    def __init__(
        self,
        model: BDH | None = None,
        block_on_memory: bool = False,
        risk_threshold: float = RISK_THRESHOLD,
        weights: dict[str, float] | None = None,
        memory_threshold: float = MEMORY_THRESHOLD,
    ):
        self.model = model
        self.block_on_memory = block_on_memory
        self.risk_threshold = risk_threshold
        self.weights = dict(WEIGHTS if weights is None else weights)
        self.memory_threshold = memory_threshold

    def clear(
        self,
        text: str,
        protected_values: list[str] = (),
        case: list[tuple[int, int]] | None = None,
        items: list[ProtectedItem] = (),
        system: str = "",
    ) -> GateResult:
        """Generalise until the combination risk is low, then run every layer.

        ``system`` is checked too, because everything that leaves is checked.
        """
        generalised: list[str] = []
        rounds = 0
        score, kinds = risk_score(text, self.weights)
        while score >= self.risk_threshold and rounds < MAX_ROUNDS:
            rounds += 1
            text, changed = generalise_quasi(text, rounds)
            generalised += [found.kind for found in changed]
            score, kinds = risk_score(text, self.weights)
        needs_review = score >= self.risk_threshold

        checked = f"{system}\n{text}" if system else text
        layers = [
            format_layer(checked),
            lexicon_layer(checked, list(protected_values)),
            Layer("combination", not needs_review, kinds if needs_review else [], score),
        ]
        memory = self.memory_layer(text, case, list(items))
        layers.append(memory)
        allowed = all(layer.passed for layer in layers[:3]) and (
            memory.passed or not self.block_on_memory
        )
        return GateResult(allowed, text, rounds, needs_review, layers, sorted(set(generalised)))

    def memory_layer(
        self, text: str, case: list[tuple[int, int]] | None, items: list[ProtectedItem]
    ) -> Layer:
        if self.model is None or not case or not items:
            return Layer("memory", True, [], None)
        scores = memory_scores(self.model, case, items, text.encode("utf-8"))
        flagged = sorted(
            {item.kind for item, score in zip(items, scores) if score >= self.memory_threshold}
        )
        return Layer("memory", not flagged, flagged, round(max(scores), 3))


def protected_items(text: str) -> tuple[list[ProtectedItem], list[str]]:
    """Find a case's protected items as byte spans, and its direct values."""
    items, values = [], []
    for found in identify(text):
        start = len(text[: found.start].encode("utf-8"))
        end = start + len(found.text.encode("utf-8"))
        items.append(ProtectedItem(found.kind, start, end))
        if found.kind in DIRECT:
            values.append(found.text)
    return items, values


__all__ = ["GateResult", "Layer", "LeakGate", "ProtectedItem", "protected_items"]
