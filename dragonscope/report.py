"""Measure whether forgetting worked, on your own conversations and questions.

    case = ds.Case(
        text="U: My code is 7341.\\nA: Noted.\\nU: My town is Galle.\\nA: Noted.\\n",
        forget=["7341"],
        probes=[
            ds.Probe("U: What is my code?\\nA: ", expect="7341", about="forgotten"),
            ds.Probe("U: What is my town?\\nA: ", expect="Galle", about="kept"),
        ],
    )
    print(ds.forgetting_report(model, [case]).to_markdown())

The report compares ``subtract`` with ``rebuild``, which is the reference: rereading
without the bytes is, by definition, what perfect forgetting of them looks like. It
tests only the probes it is given, on this model, so it is evidence and not a proof.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dragonscope.toolkit import MemoryCapture, capture
from dragonscope.vendor.bdh.bdh import BDH

# Working pass marks. Change them through forgetting_report's arguments.
MAX_FORGOTTEN_RECALL = 0.01
MAX_KEPT_DROP = 0.05
MIN_BASELINE_RECALL = 0.9
CONDITIONS = ("before", "subtract", "rebuild")


@dataclass
class Probe:
    """A prompt, and the text that counts as recalling the answer."""

    prompt: str
    expect: str
    about: str = "forgotten"  # "forgotten" or "kept"


@dataclass
class Case:
    text: str
    forget: list[str]
    probes: list[Probe]


@dataclass
class ForgettingReport:
    cases: int
    probes: int
    recall: dict[str, dict[str, float | None]]  # condition -> forgotten or kept -> share
    seconds: dict[str, float]  # method -> mean time to forget
    residual: dict[str, float]  # method -> mean distance from never reading the bytes
    residual_per_round: list[float]  # For subtract.
    replies_match_rebuild: float | None  # Share of subtract's replies equal to rebuild's.
    next_byte_distance: float | None  # Mean total variation from rebuild, 0 to 1.
    checks: dict[str, bool]
    samples: list[dict]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict:
        return {
            "cases": self.cases,
            "probes": self.probes,
            "passed": self.passed,
            "checks": self.checks,
            "recall": self.recall,
            "seconds": self.seconds,
            "residual": self.residual,
            "residual_per_round": self.residual_per_round,
            "replies_match_rebuild": self.replies_match_rebuild,
            "next_byte_distance": self.next_byte_distance,
            "samples": self.samples,
        }

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    def to_markdown(self) -> str:
        def percent(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.1%}"

        lines = [
            f"# Forgetting report: {'PASS' if self.passed else 'FAIL'}",
            "",
            f"{self.cases} conversations and {self.probes} probes.",
            "",
            "| Measure | Before | `subtract` | `rebuild` (reference) |",
            "|---|---|---|---|",
            "| Recall of forgotten values | "
            + " | ".join(percent(self.recall[c]["forgotten"]) for c in CONDITIONS)
            + " |",
            "| Recall of kept values | "
            + " | ".join(percent(self.recall[c]["kept"]) for c in CONDITIONS)
            + " |",
            f"| Time to forget | n/a | {self.seconds['subtract'] * 1000:.1f} ms | {self.seconds['rebuild'] * 1000:.1f} ms |",
            f"| Distance from never reading it | n/a | {percent(self.residual['subtract'])} | {percent(self.residual['rebuild'])} |",
            "",
            f"Replies from `subtract` matching `rebuild`: {percent(self.replies_match_rebuild)}.",
            f"Next-byte distance from `rebuild`: {percent(self.next_byte_distance)}.",
            "",
            "| Check | Result |",
            "|---|---|",
        ]
        lines += [f"| {name} | {'pass' if ok else 'FAIL'} |" for name, ok in self.checks.items()]
        lines += [
            "",
            "This covers these probes on this model. It is evidence, not a proof.",
        ]
        return "\n".join(lines)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _ask(session: MemoryCapture, probe: Probe, max_reply: int):
    """Ask from a copy of the session, returning the reply and its first probabilities."""
    trial = session.fork()
    trial.read(probe.prompt)
    first = trial.next_byte_probabilities()
    return trial.generate(max_reply), first


def forgetting_report(
    model: BDH,
    cases: Sequence[Case],
    max_reply: int = 70,
    max_forgotten_recall: float = MAX_FORGOTTEN_RECALL,
    max_kept_drop: float = MAX_KEPT_DROP,
    min_baseline_recall: float = MIN_BASELINE_RECALL,
) -> ForgettingReport:
    """Run every probe before forgetting, after ``subtract`` and after ``rebuild``."""
    if not cases:
        raise ValueError("a report needs at least one case")
    scores = {c: {"forgotten": [], "kept": []} for c in CONDITIONS}
    seconds = {"subtract": [], "rebuild": []}
    residual = {"subtract": [], "rebuild": []}
    per_round: list[list[float]] = []
    matches: list[float] = []
    distances: list[float] = []
    samples: list[dict] = []
    probes = 0

    for case in cases:
        before = capture(model)
        before.read(case.text)
        sessions = {"subtract": before.fork(), "rebuild": before.fork()}
        for method, session in sessions.items():
            total = 0.0
            last = None
            # Remove all but the last value by subtracting, as rebuild only needs to run once.
            for value in case.forget:
                last = session.forget(value, "subtract", measure=False)
                total += last.seconds
            if method == "rebuild":
                started = time.perf_counter()
                session.rebuild()
                total += time.perf_counter() - started
            seconds[method].append(total)
            measured = session.residual()
            residual[method].append(measured["total"])
            if method == "subtract":
                per_round.append(measured["per_round"])

        for probe in case.probes:
            probes += 1
            replies = {}
            first = {}
            replies["before"], first["before"] = _ask(before, probe, max_reply)
            for method, session in sessions.items():
                replies[method], first[method] = _ask(session, probe, max_reply)
            for condition in CONDITIONS:
                scores[condition][probe.about].append(float(probe.expect in replies[condition]))
            matches.append(float(replies["subtract"] == replies["rebuild"]))
            distances.append(float((first["subtract"] - first["rebuild"]).abs().sum() / 2))
            if len(samples) < 10:
                # Forgotten values can appear in the "before" reply, so it is not kept.
                samples.append(
                    {
                        "prompt": probe.prompt,
                        "about": probe.about,
                        "subtract": replies["subtract"],
                        "rebuild": replies["rebuild"],
                    }
                )

    recall = {
        condition: {about: _mean(values) for about, values in by_about.items()}
        for condition, by_about in scores.items()
    }
    checks: dict[str, bool] = {}
    baseline = recall["before"]["forgotten"]
    if baseline is not None:
        checks["Probes could recall the value before forgetting"] = baseline >= min_baseline_recall
        for method in ("subtract", "rebuild"):
            checks[f"`{method}`: forgotten value no longer recalled"] = (
                recall[method]["forgotten"] <= max_forgotten_recall
            )
    if recall["before"]["kept"] is not None:
        for method in ("subtract", "rebuild"):
            checks[f"`{method}`: other values kept"] = (
                recall["before"]["kept"] - recall[method]["kept"] <= max_kept_drop
            )
    return ForgettingReport(
        cases=len(cases),
        probes=probes,
        recall=recall,
        seconds={method: _mean(values) for method, values in seconds.items()},
        residual={method: _mean(values) for method, values in residual.items()},
        residual_per_round=[_mean(column) for column in zip(*per_round)],
        replies_match_rebuild=_mean(matches),
        next_byte_distance=_mean(distances),
        checks=checks,
        samples=samples,
    )
