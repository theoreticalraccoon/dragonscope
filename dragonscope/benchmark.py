"""Compare targeted forgetting with rebuilding, across conversation lengths.

A developer can always delete the text and reread it, which is perfect forgetting by
definition. So the targeted edit has to answer two questions: how close does it get,
and how much cheaper is it?

Run: python -m dragonscope.benchmark --model models/best_model.pt --out docs/benchmark
Add ``--scale`` to measure cost and memory at the published BDH size instead.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from random import Random

import torch

from dragonscope.data import ACKNOWLEDGEMENTS, ANSWERS, QUESTIONS, SMALL_TALK, STATEMENTS, random_card, split_names
from dragonscope.report import Case, Probe, forgetting_report
from dragonscope.toolkit import capture
from dragonscope.train import load_model
from dragonscope.vendor.bdh.bdh import BDH, BDHConfig

LENGTHS = (100, 200, 400, 800, 1600, 3200)
# The model was trained on conversations of up to 512 bytes.
TRAINED_LENGTH = 512


def build_conversation(rng: Random, length: int) -> tuple[str, str, str]:
    """Return a conversation of about ``length`` bytes that opens with a card."""
    first_names, last_names, branches = split_names("train")
    card = random_card(rng)
    branch = rng.choice(branches)
    text = f"U: {STATEMENTS['card'][0].format(card)}\nA: {rng.choice(ACKNOWLEDGEMENTS)}\n"
    text += f"U: {STATEMENTS['branch'][0].format(branch)}\nA: {rng.choice(ACKNOWLEDGEMENTS)}\n"
    while len(text) < length:
        question, reply = rng.choice(SMALL_TALK)
        text += f"U: {question}\nA: {reply}\n"
    return text, card, branch


def measure_length(model, length: int, cases: int, seed: int) -> dict:
    rng = Random(seed + length)
    built = [build_conversation(rng, length) for _ in range(cases)]
    report = forgetting_report(
        model,
        [
            Case(
                text,
                [card],
                [
                    Probe(f"U: {QUESTIONS['card'][0]}\nA: ", card, "forgotten"),
                    Probe(f"U: {QUESTIONS['branch'][0]}\nA: ", ANSWERS["branch"].format(branch), "kept"),
                ],
            )
            for text, card, branch in built
        ],
    )

    # Time each method on its own, repeatedly, as the report's timing is only one pass.
    subtract, rebuild = [], []
    for text, card, _ in built:
        base = capture(model)
        base.memory.observer = None
        base.memory.read(text.encode("ascii"))
        for _ in range(3):
            trial = base.memory.fork()
            start, end = trial.find_positions(card.encode("ascii"))[0]
            started = time.perf_counter()
            trial.forget(start, end, "subtract")
            subtract.append(time.perf_counter() - started)

            trial = base.memory.fork()
            started = time.perf_counter()
            trial.forget(start, end, "replay")
            rebuild.append(time.perf_counter() - started)
    ledger_bytes = base.memory.ledger_bytes()
    subtract_ms = statistics.median(subtract) * 1000
    rebuild_ms = statistics.median(rebuild) * 1000
    return {
        "length": len(built[0][0]),
        "within_trained_length": len(built[0][0]) <= TRAINED_LENGTH,
        "cases": cases,
        "secret_bytes": len(built[0][1]),
        "subtract_ms": subtract_ms,
        "rebuild_ms": rebuild_ms,
        "speedup": rebuild_ms / subtract_ms,
        "residual": report.residual["subtract"],
        "residual_per_round": report.residual_per_round,
        "rebuild_residual": report.residual["rebuild"],
        "recall_before": report.recall["before"],
        "recall_subtract": report.recall["subtract"],
        "recall_rebuild": report.recall["rebuild"],
        "replies_match_rebuild": report.replies_match_rebuild,
        "next_byte_distance": report.next_byte_distance,
        "ledger_mb": ledger_bytes / 1e6,
    }


def measure_erased_size(model, seed: int) -> list[dict]:
    """How does the cost change with how much is erased, at a fixed length?"""
    rng = Random(seed)
    text = build_conversation(rng, 1600)[0]
    base = capture(model)
    base.memory.observer = None
    base.memory.read(text.encode("ascii"))
    rows = []
    for size in (5, 20, 80, 320):
        times = []
        for _ in range(5):
            trial = base.memory.fork()
            started = time.perf_counter()
            trial.forget(100, 100 + size, "subtract")
            times.append(time.perf_counter() - started)
        rows.append({"bytes_erased": size, "subtract_ms": statistics.median(times) * 1000})
    return rows


def measure_scale(lengths=(256, 512, 1024), seed: int = 20261102) -> dict:
    """Cost and memory at the published BDH default size, with random weights.

    Random weights fire more neurons than trained ones, so the compact ledger saves
    less here than it would on a trained model.
    """
    torch.manual_seed(seed)
    config = BDHConfig()
    model = BDH(config).eval()
    rng = Random(seed)
    rows = []
    for length in lengths:
        text, card, _ = build_conversation(rng, length)
        data = text.encode("ascii")[:length]
        if card.encode("ascii") not in data:
            raise ValueError("the card must be inside the measured text")
        sizes = {}
        for mode in ("exact", "compact"):
            memory = capture(model, ledger=mode)
            memory.memory.observer = None
            started = time.perf_counter()
            memory.memory.read(data)
            read_seconds = time.perf_counter() - started
            sizes[mode] = memory.ledger_bytes / len(data)
            if mode == "exact":
                base = memory
                per_byte_ms = read_seconds / len(data) * 1000
        start, end = base.memory.find_positions(card.encode("ascii"))[0]
        subtract, rebuild = [], []
        for _ in range(3):
            trial = base.memory.fork()
            started = time.perf_counter()
            trial.forget(start, end, "subtract")
            subtract.append(time.perf_counter() - started)
            trial = base.memory.fork()
            started = time.perf_counter()
            trial.forget(start, end, "replay")
            rebuild.append(time.perf_counter() - started)
        rows.append(
            {
                "length": len(data),
                "read_ms_per_byte": per_byte_ms,
                "ledger_kb_per_byte_exact": sizes["exact"] / 1e3,
                "ledger_kb_per_byte_compact": sizes["compact"] / 1e3,
                "subtract_ms": statistics.median(subtract) * 1000,
                "rebuild_ms": statistics.median(rebuild) * 1000,
            }
        )
        del base, trial
    rho_bytes = config.n_layer * config.n_embd * config.mlp_internal_dim_multiplier * config.n_embd * 4
    return {
        "config": {
            "n_layer": config.n_layer,
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "neurons": config.mlp_internal_dim_multiplier * config.n_embd,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "weights": "random",
        "threads": torch.get_num_threads(),
        "rho_mb": rho_bytes / 1e6,
        "secret_bytes": len(card),
        "lengths": rows,
    }


def run(model, lengths=LENGTHS, cases: int = 20, seed: int = 20261102, progress: bool = False) -> dict:
    results = []
    for length in lengths:
        results.append(measure_length(model, length, cases, seed))
        if progress:
            row = results[-1]
            print(
                f"{row['length']:>5} bytes: subtract {row['subtract_ms']:7.1f} ms, "
                f"rebuild {row['rebuild_ms']:8.1f} ms, residual {row['residual']:.2%}, "
                f"recall before {row['recall_before']['forgotten']:.0%} "
                f"after {row['recall_subtract']['forgotten']:.0%}/{row['recall_rebuild']['forgotten']:.0%}, "
                f"kept {row['recall_before']['kept']:.0%}->{row['recall_subtract']['kept']:.0%}/{row['recall_rebuild']['kept']:.0%}, "
                f"same replies {row['replies_match_rebuild']:.0%}, tv {row['next_byte_distance']:.3f}",
                flush=True,
            )
    return {
        "threads": torch.get_num_threads(),
        "lengths": results,
        "erased_size": measure_erased_size(model, seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument("--out", type=Path, default=Path("docs/benchmark"))
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--scale", action="store_true", help="measure at the published BDH size")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.scale:
        scale = measure_scale()
        (args.out / "scale.json").write_text(json.dumps(scale, indent=1), encoding="utf-8")
        print(json.dumps(scale, indent=1))
        return
    results = run(load_model(args.model), cases=args.cases, progress=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results["erased_size"]))


if __name__ == "__main__":
    main()
