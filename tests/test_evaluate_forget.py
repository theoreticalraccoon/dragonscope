import json
from random import Random

import torch

from dragonscope.engine import SessionMemory
from dragonscope.evaluate_forget import (
    Case,
    build_case,
    digit_match,
    forget_every_copy,
    format_summary,
    leak_scores,
    load_results,
    longest_shared_run,
    run,
    summarise,
)
from dragonscope.model_config import demo_config
from dragonscope.vendor.bdh.bdh import BDH


def small_model():
    torch.manual_seed(0)
    return BDH(demo_config()).eval()


def test_case_spans_point_at_values():
    case = build_case(Random(1), "card")
    for field, (start, end) in case.spans.items():
        assert case.text[start:end].decode() == case.values[field]


def test_leak_scores_catch_reformatted_secret():
    case = Case(b"", {"card": "4111 1111 1111 1111"}, {}, "card", "5000 0000 0000 0000")
    scores = leak_scores("Your card number is 4111-1111-1111-1111.", case)
    assert scores["exact_leak"] == 1.0
    assert scores["digit_match"] == 1.0
    assert scores["control_digit_match"] == 0.0


def test_digit_scores_on_partial_overlap():
    assert digit_match("12xx", "1234") == 0.5
    assert longest_shared_run("9923499", "1234") == 3
    assert longest_shared_run("no digits", "1234") == 0


def test_forget_every_copy_removes_all_occurrences():
    session = SessionMemory(small_model())
    session.read(b"A 1234 B 1234 C")
    forget_every_copy(session, "1234", "subtract")
    assert session.text() == b"A  B  C"
    assert session.find_positions(b"1234") == []


def test_fork_does_not_share_memory():
    session = SessionMemory(small_model())
    session.read(b"hello")
    fork = session.fork()
    fork.read(b" world")
    fork.forget(0, 2)
    assert session.text() == b"hello"
    assert session.position == 5
    assert not torch.equal(session.rho, fork.rho)


def test_full_evaluation_runs_on_an_untrained_model():
    results = run(small_model(), cases=1, seed=3)
    summary = summarise(results)
    assert summary["baseline"]["direct"]["exact_leak"] == 0.0
    assert summary["replay"]["residual"] < 1e-5
    assert summary["subtract"]["residual"] > 0.01
    assert "echoed" in summary["replay"]
    assert "replay" in format_summary(summary)


def test_rerun_resumes_and_matches_an_uninterrupted_run(tmp_path):
    model = small_model()
    whole = run(model, cases=2, seed=5)
    out = tmp_path / "shard.jsonl"
    run(model, cases=1, seed=5, out=out)
    with open(out, "a", encoding="utf-8") as file:
        file.write('{"cut short')
    resumed = run(model, cases=2, seed=5, out=out)
    assert resumed == json.loads(json.dumps(whole))
    assert len(load_results(out)) == 2
