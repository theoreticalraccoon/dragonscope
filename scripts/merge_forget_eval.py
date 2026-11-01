"""Combine shards from dragonscope.evaluate_forget into one summary.

Run: ``python scripts/merge_forget_eval.py runs/forget_eval_500.json runs/shards/*.jsonl``
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dragonscope.evaluate_forget import format_summary, load_results, summarise


def main() -> None:
    output = Path(sys.argv[1])
    results = []
    for shard in sys.argv[2:]:
        results.extend(load_results(Path(shard)))
    summary = summarise(results)
    output.write_text(
        json.dumps({"cases": len(results), "summary": summary}, indent=1),
        encoding="utf-8",
    )
    print(f"cases: {len(results)}")
    print(format_summary(summary))

    for target in ("card", "nic"):
        subset = [r for r in results if r["target"] == target]
        if subset:
            print(f"\ntarget = {target} ({len(subset)} cases)")
            print(format_summary(summarise(subset)))


if __name__ == "__main__":
    main()
