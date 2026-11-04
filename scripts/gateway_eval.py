"""Measure the gateway on synthetic data: leaks, false blocks, the memory check and time.

Run: python scripts/gateway_eval.py [--cases 300] [--model models/best_model.pt]

Results go to docs/gateway/results.json, failures included. The synthetic data
shares word lists with the detector, except the "unlisted" names, so the listed
rows are optimistic; the unlisted and no-cue rows show what the rules miss.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from random import Random

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dragonscope.gateway import Gateway, LeakGate  # noqa: E402
from dragonscope.gateway import samples  # noqa: E402
from dragonscope.gateway.gate import memory_scores, protected_items  # noqa: E402
from dragonscope.model_config import demo_config  # noqa: E402
from dragonscope.train import load_model  # noqa: E402
from dragonscope.vendor.bdh.bdh import BDH  # noqa: E402


def _present(kind: str, value: str, text: str) -> bool:
    if kind == "age_90":
        # "90+" is the generalised form, so only the exact age counts.
        return re.search(r"\b" + re.escape(value) + r"\b(?!\+)", text) is not None
    return value in text


def leaks(gateway: Gateway, cases: list[samples.Sample]) -> dict:
    labelled, left, leaked, blocked = Counter(), Counter(), Counter(), 0
    by_variant = Counter()
    by_variant_leaked = Counter()
    seconds = []
    for index, sample in enumerate(cases):
        ref = f"eval-{index}"
        gateway.open_case(ref, sample.text)
        draft = gateway.submit(ref, sample.task, "evaluator")
        seconds.append(draft.seconds)
        cleared = draft.sent_text
        if cleared is None:
            blocked += 1
        variant = "unlisted name" if any(
            v.split()[0] in samples.UNLISTED_FIRST for k, v in sample.labels if k == "name"
        ) else "listed name"
        if not sample.text.startswith(("Customer:", "Patient:")):
            variant = "no cue"
        for kind, value in sample.labels:
            labelled[kind] += 1
            by_variant[variant] += 1
            if cleared is not None and _present(kind, value, cleared):
                leaked[kind] += 1
                by_variant_leaked[variant] += 1
        gateway.erase_case(ref)
    total = sum(labelled.values())
    return {
        "cases": len(cases),
        "identifiers": total,
        "leaked": sum(leaked.values()),
        "leak_rate": round(sum(leaked.values()) / total, 4),
        "blocked_requests": blocked,
        "by_kind": {
            kind: {"labelled": labelled[kind], "leaked": leaked[kind]} for kind in sorted(labelled)
        },
        "by_variant": {
            variant: {"identifiers": by_variant[variant], "leaked": by_variant_leaked[variant]}
            for variant in sorted(by_variant)
        },
        "seconds_mean": round(statistics.mean(seconds), 3),
        "seconds_max": round(max(seconds), 3),
    }


def false_blocks(gateway: Gateway, count: int, rng: Random) -> dict:
    blocked = 0
    for index in range(count):
        sample = samples.clean_case(rng)
        gateway.open_case(f"clean-{index}", sample.text)
        if gateway.submit(f"clean-{index}", sample.task, "evaluator").status == "blocked":
            blocked += 1
        gateway.erase_case(f"clean-{index}")
    return {"clean_requests": count, "blocked": blocked, "rate": round(blocked / count, 4)}


def memory_check(model: BDH, gate: LeakGate, seeds: int) -> dict:
    """Score paraphrases of a protected occupation against clean controls."""
    paraphrase_scores, control_scores = [], []
    rules_caught = 0
    rng = Random(7)
    for _ in range(seeds):
        case, paraphrases, controls = samples.red_team(rng)
        items, values = protected_items(case)
        pairs = list(enumerate(case.encode("utf-8")))
        for text, bucket in [(p, paraphrase_scores) for p in paraphrases] + [
            (c, control_scores) for c in controls
        ]:
            scores = memory_scores(model, pairs, items, text.encode("utf-8"))
            bucket.append(max(scores))
        for text in paraphrases:
            result = gate.clear(text, values)
            if not all(layer.passed for layer in result.layers[:3]):
                rules_caught += 1
    pairs_compared = [(p, c) for p in paraphrase_scores for c in control_scores]
    separation = sum(p > c for p, c in pairs_compared) / len(pairs_compared)
    threshold = gate.memory_threshold
    return {
        "paraphrases": len(paraphrase_scores),
        "controls": len(control_scores),
        "rules_layers_1_to_3_caught": rules_caught,
        "memory_threshold": threshold,
        "memory_caught": sum(s >= threshold for s in paraphrase_scores),
        "memory_false_flags": sum(s >= threshold for s in control_scores),
        "paraphrase_score_mean": round(statistics.mean(paraphrase_scores), 3),
        "control_score_mean": round(statistics.mean(control_scores), 3),
        "separation": round(separation, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--clean", type=int, default=100)
    parser.add_argument("--red-team-seeds", type=int, default=20)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "best_model.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "gateway" / "results.json")
    args = parser.parse_args()

    if args.model.exists():
        model, model_name = load_model(args.model), args.model.name
    else:
        torch.manual_seed(0)
        model, model_name = BDH(demo_config()).eval(), "random weights"
    gate = LeakGate(model)
    gateway = Gateway(model, gate=gate)
    started = time.perf_counter()
    results = {
        "model": model_name,
        "connector": gateway.connector.name,
        "leaks": leaks(gateway, samples.build(args.cases, seed=0)),
        "false_blocks": false_blocks(gateway, args.clean, Random(1)),
        "memory_check": memory_check(model, gate, args.red_team_seeds),
    }
    results["seconds_total"] = round(time.perf_counter() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
