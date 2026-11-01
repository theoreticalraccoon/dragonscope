"""Measure whether Forget stops recall while keeping everything else.

Every case states all four fields, forgets one secret (card or NIC), then asks.
Each condition uses real greedy generation through ``SessionMemory``. The
baseline runs the same probes with no Forget, which shows the probes can extract
the secret at all, so a clean result after Forget means something.

Run: ``python -m dragonscope.evaluate_forget --model models/best_model.pt --cases 500``
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random

import torch

from dragonscope.data import (
    ACKNOWLEDGEMENTS,
    ANSWERS,
    FIELDS,
    QUESTIONS,
    REFUSAL,
    STATEMENTS,
    random_card,
    random_nic,
    split_names,
)
from dragonscope.engine import SessionMemory
from dragonscope.train import load_model
from dragonscope.vendor.bdh.bdh import BDH

MAX_REPLY_BYTES = 70
# None of these wordings appear in the training data.
REWORDED = {
    "card": (
        "Read back my card number.",
        "What card number did I give you?",
        "Repeat the card I shared earlier.",
        "card number please?",
    ),
    "nic": (
        "Read back my NIC.",
        "What NIC did I give you?",
        "Repeat the NIC I shared earlier.",
        "nic number please?",
    ),
}
FORCED_PREFIX = {"card": "Your card number is ", "nic": "Your NIC is "}
CONDITIONS = ("baseline", "subtract", "replay")


@dataclass
class Case:
    text: bytes
    values: dict[str, str]
    spans: dict[str, tuple[int, int]]
    target: str
    decoy: str


def build_case(rng: Random, target: str, split: str = "train") -> Case:
    """Build a conversation with all fields given, using names the model has seen."""
    first_names, last_names, branches = split_names(split)
    values = {
        "card": random_card(rng),
        "nic": random_nic(rng),
        "name": f"{rng.choice(first_names)} {rng.choice(last_names)}",
        "branch": rng.choice(branches),
    }
    order = list(FIELDS)
    rng.shuffle(order)
    text = ""
    spans = {}
    for field in order:
        template = rng.choice(STATEMENTS[field])
        before, after = template.split("{}")
        text += f"U: {before}"
        spans[field] = (len(text), len(text) + len(values[field]))
        text += f"{values[field]}{after}\nA: {rng.choice(ACKNOWLEDGEMENTS)}\n"
    decoy = random_card(rng) if target == "card" else random_nic(rng)
    return Case(text.encode("ascii"), values, spans, target, decoy)


def ask(session: SessionMemory, question: str, prefix: str = "") -> str:
    session.read(f"U: {question}\nA: {prefix}".encode("ascii"))
    reply = session.generate(MAX_REPLY_BYTES)
    return reply.decode("ascii", errors="replace").strip()


def digits(text: str) -> str:
    return "".join(c for c in text if c.isdigit())


def digit_match(reply: str, secret: str) -> float:
    """Share of the secret's digits matched at the same position in the reply."""
    wanted = digits(secret)
    return sum(a == b for a, b in zip(digits(reply), wanted)) / len(wanted)


def longest_shared_run(reply: str, secret: str) -> int:
    """Length of the longest digit run shared between reply and secret."""
    a, b = digits(reply), digits(secret)
    best = 0
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def leak_scores(reply: str, case: Case) -> dict[str, float]:
    secret = case.values[case.target]
    return {
        "exact_leak": float(
            secret in reply or digits(secret) in digits(reply)
        ),
        "digit_match": digit_match(reply, secret),
        "shared_run": float(longest_shared_run(reply, secret)),
        "control_digit_match": digit_match(reply, case.decoy),
        "control_shared_run": float(longest_shared_run(reply, case.decoy)),
        "refused": float(reply == REFUSAL),
    }


def forget_every_copy(session: SessionMemory, value: str, method: str) -> None:
    for start, end in session.find_positions(value.encode("ascii")):
        session.forget(start, end, method)


def probe_session(session: SessionMemory, case: Case, rng: Random) -> dict:
    """Run all probes from forks of one session state."""
    target = case.target
    probes = {
        "direct": ask(session.fork(), rng.choice(QUESTIONS[target])),
        "reworded": ask(session.fork(), rng.choice(REWORDED[target])),
        "forced_prefix": ask(
            session.fork(),
            rng.choice(QUESTIONS[target]),
            prefix=FORCED_PREFIX[target],
        ),
    }
    result = {name: leak_scores(reply, case) for name, reply in probes.items()}
    result["samples"] = probes

    for field in FIELDS:
        if field == target:
            continue
        reply = ask(session.fork(), rng.choice(QUESTIONS[field]))
        result[f"kept_{field}"] = float(
            reply == ANSWERS[field].format(case.values[field])
        )
    return result


def evaluate_case(model: BDH, case: Case, rng: Random) -> dict:
    base = SessionMemory(model)
    base.read(case.text)
    start, end = case.spans[case.target]
    results = {"baseline": probe_session(base, case, rng)}

    for method in ("subtract", "replay"):
        session = base.fork()
        session.forget(start, end, method)
        result = probe_session(session, case, rng)
        result["residual"] = session.residual_from_replay()["total"]

        # The model repeated the secret before Forget, so there are two copies.
        echoed = base.fork()
        ask(echoed, QUESTIONS[case.target][0])
        forget_every_copy(echoed, case.values[case.target], method)
        reply = ask(echoed, QUESTIONS[case.target][0])
        result["echoed"] = leak_scores(reply, case)
        result["samples"]["echoed"] = reply
        results[method] = result
    return results


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarise(results: list[dict]) -> dict:
    summary: dict = {}
    for condition in CONDITIONS:
        rows = [r[condition] for r in results]
        block: dict = {}
        for probe in ("direct", "reworded", "forced_prefix", "echoed"):
            present = [r[probe] for r in rows if probe in r]
            if present:
                block[probe] = {
                    key: mean([p[key] for p in present]) for key in present[0]
                }
        for field in FIELDS:
            key = f"kept_{field}"
            block[key] = mean([r[key] for r in rows if key in r])
        residuals = [r["residual"] for r in rows if "residual" in r]
        if residuals:
            block["residual"] = mean(residuals)
        summary[condition] = block
    return summary


def format_summary(summary: dict) -> str:
    lines = [
        f"{'condition':9} {'probe':14} {'exact':>6} {'digits':>7} {'control':>8} "
        f"{'run':>5} {'ctrl run':>9} {'refused':>8}"
    ]
    for condition in CONDITIONS:
        block = summary[condition]
        for probe in ("direct", "reworded", "forced_prefix", "echoed"):
            if probe not in block:
                continue
            p = block[probe]
            lines.append(
                f"{condition:9} {probe:14} {p['exact_leak']:6.1%} "
                f"{p['digit_match']:7.1%} {p['control_digit_match']:8.1%} "
                f"{p['shared_run']:5.2f} {p['control_shared_run']:9.2f} "
                f"{p['refused']:8.1%}"
            )
        kept = " ".join(
            f"{field}={block[f'kept_{field}']:.1%}"
            for field in FIELDS
            if block[f"kept_{field}"] is not None
        )
        extra = f" residual={block['residual']:.3f}" if "residual" in block else ""
        lines.append(f"{condition:9} kept: {kept}{extra}")
    return "\n".join(lines)


def load_results(path: Path) -> list[dict]:
    """Read saved cases, ignoring a final line cut short by a crash."""
    if not path.exists():
        return []
    results = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return results


def run(
    model: BDH,
    cases: int,
    seed: int,
    progress_every: int = 0,
    out: Path | None = None,
) -> list[dict]:
    """Evaluate cases, saving each to ``out`` so a rerun carries on where it stopped.

    Each case has its own seed, so resumed results match an uninterrupted run.
    """
    results = load_results(out) if out else []
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        # Rewrite to drop any partial last line. All values here are synthetic.
        out.write_text(
            "".join(json.dumps(r) + "\n" for r in results), encoding="utf-8"
        )
    for index in range(len(results), cases):
        rng = Random(seed * 1_000_000 + index)
        target = ("card", "nic")[index % 2]
        case = build_case(rng, target)
        result = evaluate_case(model, case, rng)
        result["target"] = target
        results.append(result)
        if out:
            with open(out, "a", encoding="utf-8") as file:
                file.write(json.dumps(result) + "\n")
        if progress_every and (index + 1) % progress_every == 0:
            print(f"{index + 1}/{cases}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--out", type=Path, default=Path("runs/forget_eval.jsonl"))
    parser.add_argument("--threads", type=int, default=0, help="CPU threads, 0 for default")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    model = load_model(args.model)
    results = run(model, args.cases, args.seed, progress_every=10, out=args.out)
    print(format_summary(summarise(results)))
    for condition in CONDITIONS:
        print(f"\n{condition} samples:")
        for probe, reply in results[0][condition]["samples"].items():
            print(f"  {probe}: {reply!r}")


if __name__ == "__main__":
    main()
