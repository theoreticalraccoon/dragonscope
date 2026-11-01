"""Gate tests for the stateful engine and Forget v1, on untrained models."""

import pytest
import torch

from dragonscope.engine import SessionMemory
from dragonscope.model_config import demo_config
from dragonscope.vendor.bdh.bdh import BDH, BDHConfig

TEXT = (
    b"U: Hi, my card number is 4111 1111 1111 1111.\n"
    b"A: Thanks, noted.\nU: My branch is Kandy.\nA: Noted.\n"
    b"U: What is my card number?\nA: "
)
SECRET_START = TEXT.index(b"4111")
SECRET_END = SECRET_START + len(b"4111 1111 1111 1111")


def make_model(config=None, seed=0):
    torch.manual_seed(seed)
    return BDH(config or demo_config()).eval()


@pytest.fixture(scope="module")
def model():
    return make_model()


@torch.no_grad()
@pytest.mark.parametrize(
    "config",
    [
        demo_config(),
        BDHConfig(n_layer=2, n_embd=64, n_head=2, mlp_internal_dim_multiplier=8),
        BDHConfig(n_layer=6, n_embd=256, n_head=4, mlp_internal_dim_multiplier=128),
    ],
    ids=["demo", "tiny", "upstream-default"],
)
def test_stateful_logits_match_full_forward(config):
    model = make_model(config)
    data = TEXT[:48] if config.n_embd == 256 else TEXT
    full, _ = model(torch.tensor([list(data)]))
    stateful = SessionMemory(model).read(data)
    assert (stateful - full[0]).abs().max().item() < 1e-3


def test_split_reads_match_a_single_read(model):
    whole = SessionMemory(model).read(TEXT)
    session = SessionMemory(model)
    parts = [session.read(TEXT[:20]), session.read(TEXT[20:])]
    assert (torch.cat(parts) - whole).abs().max().item() < 1e-5


def test_ledger_sums_to_rho(model):
    session = SessionMemory(model)
    session.read(TEXT)
    assert (session.ledger_sum() - session.rho).abs().max().item() < 1e-3


def test_rho_changes_when_a_byte_is_read(model):
    session = SessionMemory(model)
    session.read(b"A")
    assert session.rho.abs().sum().item() > 0


def test_forgetting_final_bytes_by_subtraction_is_exact(model):
    before = SessionMemory(model)
    before.read(TEXT[:SECRET_START])
    session = SessionMemory(model)
    session.read(TEXT[:SECRET_END])
    session.forget(SECRET_START, SECRET_END)
    assert (session.rho - before.rho).abs().max().item() < 1e-3


@pytest.mark.parametrize("method", ["subtract", "replay"])
def test_forget_destroys_bytes_ledger_and_pending_logits(model, method):
    session = SessionMemory(model)
    session.read(TEXT)
    erased = session.forget(SECRET_START, SECRET_END, method)
    secret_range = set(range(SECRET_START, SECRET_END))
    assert set(erased) == secret_range
    assert secret_range.isdisjoint(session.bytes)
    assert secret_range.isdisjoint(session.ledger)
    assert session.next_logits is None
    assert session.position == len(TEXT)
    with pytest.raises(RuntimeError):
        session.generate(1)


def test_replay_matches_a_session_that_never_saw_the_secret(model):
    session = SessionMemory(model)
    session.read(TEXT)
    session.forget(SECRET_START, SECRET_END, "replay")

    never_saw = SessionMemory(model)
    for position, byte in enumerate(TEXT):
        if not SECRET_START <= position < SECRET_END:
            never_saw._step(byte, position)
    assert (session.rho - never_saw.rho).abs().max().item() < 1e-5
    assert session.residual_from_replay()["total"] < 1e-5


def test_subtract_is_exact_at_layer_zero_and_reports_deeper_residual(model):
    session = SessionMemory(model)
    session.read(TEXT)
    session.forget(SECRET_START, SECRET_END, "subtract")
    residual = session.residual_from_replay()
    assert residual["per_layer"][0] < 1e-3
    # Bytes after the secret absorbed it at deeper layers, so this is the leak
    # that Monday's evaluation has to measure.
    assert max(residual["per_layer"][1:]) > residual["per_layer"][0]


def test_session_keeps_reading_after_forget(model):
    session = SessionMemory(model)
    session.read(TEXT)
    session.forget(SECRET_START, SECRET_END, "replay")
    assert session.read(b"U")[0].isfinite().all()
    assert session.position == len(TEXT) + 1


def test_forgetting_an_erased_range_again_changes_nothing(model):
    session = SessionMemory(model)
    session.read(TEXT)
    session.forget(SECRET_START, SECRET_END)
    rho = session.rho.clone()
    assert session.forget(SECRET_START, SECRET_END) == []
    assert torch.equal(session.rho, rho)


def test_unknown_method_is_rejected(model):
    with pytest.raises(ValueError):
        SessionMemory(model).forget(0, 1, "shred")
