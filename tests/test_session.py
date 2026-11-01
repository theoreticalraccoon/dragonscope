"""Session behaviour with scripted replies, since an untrained model writes noise."""

import json

import pytest
import torch

from dragonscope.data import split_names
from dragonscope.model_config import demo_config
from dragonscope.session import DemoSession, SessionError, guide
from dragonscope.vendor.bdh.bdh import BDH

CARD = "4111 1111 1111 1111"
NIC = "123456789V"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return BDH(demo_config()).eval()


@pytest.fixture
def session(model):
    return DemoSession(model)


def script(session, *replies):
    """Make the model write these replies, in order, through the real read path."""
    queue = list(replies)

    def generate(max_new_bytes, stop_byte=10):
        written = queue.pop(0).encode("ascii")
        session.memory.read(written)
        return written

    session.memory.generate = generate


def visible(session):
    return json.dumps(session.state())


def turn_text(turn):
    return "".join(segment["text"] for segment in turn["segments"])


def flagged(session, kind, source="user"):
    return next(
        f for f in session.findings if f.kind == kind and f.source == source
    )


def assert_gone(session, *secrets):
    text = session.memory.text().decode("latin-1")
    for secret in secrets:
        assert secret not in text
        assert secret not in visible(session)


def test_statement_is_flagged_and_stored(session):
    script(session, "Noted.\n")
    session.say("card", CARD)
    state = session.state()
    assert [f["kind"] for f in state["findings"]] == ["card"]
    assert state["findings"][0]["last4"] == "1111"
    assert state["findings"][0]["luhn_valid"] is True
    assert turn_text(state["turns"][0]) == f"My card number is {CARD}."
    assert any("finding" in s for s in state["turns"][0]["segments"])


def test_invalid_value_never_reaches_the_model(session):
    with pytest.raises(SessionError) as error:
        session.say("card", "4111 1111 1111 1112")
    assert "4111" not in error.value.message
    assert session.memory.position == 0


def test_only_trained_names_and_branches_are_accepted(session):
    first_names, last_names, branches = split_names("train")
    held_out_first, _, held_out_branches = split_names("val")
    with pytest.raises(SessionError):
        session.say("name", f"{held_out_first[0]} {last_names[0]}")
    with pytest.raises(SessionError):
        session.say("branch", held_out_branches[0])
    script(session, "Noted.\n", "Noted.\n")
    session.say("name", f"{first_names[0]} {last_names[0]}")
    session.say("branch", branches[0])
    assert session.findings == []


def test_guide_only_offers_what_is_accepted():
    offered = guide()
    assert offered["branches"] == split_names("train")[2]
    assert "{}" in offered["statements"]["card"]


def test_forget_erases_statement_echo_and_garbled_echo(session):
    garbled = "4711 1111 1111 5111"
    script(
        session,
        "Noted.\n",
        f"Your card number is {CARD}.\n",
        f"Your card number is {garbled}.\n",
    )
    session.say("card", CARD)
    session.ask("card")
    session.ask("card")
    assert len(session.findings) == 3

    result = session.forget(flagged(session, "card").id)

    assert result["copies_erased"] == 3
    assert_gone(session, CARD, garbled, "4711")
    assert session.findings == []
    assert session.memory.next_logits is None
    assert result["checks"] == {
        "value_in_memory_bytes": False,
        "ledger_entries_left": 0,
        "pending_logits": False,
    }


def test_forget_finds_a_copy_in_another_format(session):
    script(session, "Noted.\n", "Your card number is 4111-1111-1111-1111.\n")
    session.say("card", CARD)
    session.ask("card")
    session.forget(flagged(session, "card").id)
    assert_gone(session, CARD, "4111-1111-1111-1111")


def test_forgetting_from_the_echo_also_erases_the_statement(session):
    script(session, "Noted.\n", f"Your card number is {CARD}.\n")
    session.say("card", CARD)
    session.ask("card")
    session.forget(flagged(session, "card", source="reply").id)
    assert_gone(session, CARD)


def test_forget_keeps_other_secrets_and_their_echoes(session):
    script(
        session,
        "Noted.\n",
        "Noted.\n",
        f"Your NIC is {NIC}.\n",
        f"Your card number is {CARD}.\n",
    )
    session.say("card", CARD)
    session.say("nic", NIC)
    session.ask("nic")
    session.ask("card")
    session.forget(flagged(session, "card").id)
    assert_gone(session, CARD)
    text = session.memory.text().decode("latin-1")
    assert text.count(NIC) == 2
    assert {f.kind for f in session.findings} == {"nic"}


def test_forget_removes_a_garbled_number_of_unknown_origin(session):
    script(session, "Noted.\n", "Noted.\n", "Your NIC is 99912.\n")
    session.say("card", CARD)
    session.say("nic", NIC)
    session.ask("nic")
    session.forget(flagged(session, "card").id)
    assert_gone(session, "99912")
    assert NIC in session.memory.text().decode("latin-1")


def test_rebuild_leaves_no_residual_and_subtract_is_reported(session):
    script(session, "Noted.\n", "Noted.\n")
    session.say("card", CARD)
    session.say("nic", NIC)
    result = session.forget(flagged(session, "card").id)
    assert result["residual_after_subtract"] > 0.001
    assert result["residual_after_rebuild"] < 1e-5
    assert session.memory.residual_from_replay()["total"] < 1e-5


def test_deletion_log_holds_only_type_last4_and_time(session):
    script(session, "Noted.\n")
    session.say("card", CARD)
    result = session.forget(flagged(session, "card").id)
    assert set(result["log_entry"]) == {"type", "last4", "time"}
    assert session.deletion_log == [result["log_entry"]]
    assert session.deletion_log[0]["type"] == "card"
    assert session.deletion_log[0]["last4"] == "1111"


def test_forgotten_text_is_marked_without_its_value(session):
    script(session, "Noted.\n")
    session.say("card", CARD)
    session.forget(flagged(session, "card").id)
    segments = session.state()["turns"][0]["segments"]
    assert {"text": "", "forgotten": True} in segments
    assert turn_text(session.state()["turns"][0]) == "My card number is ."


def test_session_keeps_working_after_forget(session):
    script(session, "Noted.\n", "I don't have that on record.\n")
    session.say("card", CARD)
    session.forget(flagged(session, "card").id)
    session.ask("card")
    assert session.last_answer == {"field": "card", "refused": True}


def test_neuron_memory_covers_every_neuron_and_clears_flagged_data(session):
    script(session, "Noted.\n", "Noted.\n")
    session.say("branch", split_names("train")[2][0])
    session.say("card", CARD)
    memory = session.state()["memory"]
    neurons = session.memory.n_head * session.memory.N
    assert len(memory["strength"]) == len(memory["flagged"]) == neurons
    assert max(memory["strength"]) == 100
    assert max(memory["flagged"]) > 10
    session.forget(flagged(session, "card").id)
    assert set(session.state()["memory"]["flagged"]) == {0}


def test_empty_session_has_a_state(session):
    assert set(session.state()["memory"]["strength"]) == {0}


def test_reset_clears_everything(session):
    script(session, "Noted.\n")
    session.say("card", CARD)
    session.forget(flagged(session, "card").id)
    session.reset()
    state = session.state()
    assert state["turns"] == [] and state["findings"] == []
    assert state["deletion_log"] == []
    assert session.memory.position == 0 and session.memory.ledger == {}


def test_repeat_statement_is_refused(session):
    script(session, "Noted.\n")
    session.say("card", CARD)
    with pytest.raises(SessionError) as error:
        session.say("card", "5500 0000 0000 0004")
    assert error.value.code == "already_stated"


def test_forgotten_field_cannot_be_restated_until_reset(session):
    script(session, "Noted.\n", "Noted.\n")
    session.say("card", CARD)
    session.forget(flagged(session, "card").id)
    with pytest.raises(SessionError) as error:
        session.say("card", "5500 0000 0000 0004")
    assert error.value.code == "already_stated"
    session.reset()
    session.say("card", "5500 0000 0000 0004")


def test_activity_has_one_frame_per_byte_and_is_handed_over_once(session):
    session.say("card", CARD)
    before = session.memory.position
    frames = session.take_activity()
    assert len(frames) == before
    assert frames[0]["phase"] == "read" and frames[-1]["phase"] == "write"
    assert session.take_activity() is None
    assert session.memory.observer is None
    assert len(frames[0]["fired"]) == session.memory.n_layer
    neurons = session.memory.n_head * session.memory.N
    assert all(0 <= i < neurons for i in frames[0]["fired"][2]["i"])
    assert all(count > 0 for count in frames[0]["counts"])
    assert len(frames[0]["links"]) == session.memory.n_layer - 1
    for round_links in frames[0]["links"]:
        assert all(0 <= a < neurons and 0 <= b < neurons for a, b in round_links)
    assert set(frames[0]) == {"phase", "fired", "counts", "links", "flagged", "depends"}


def test_activity_reports_flagged_neurons_only_while_something_is_flagged(session):
    script(session, "Noted.\n", "Noted.\n", "Noted.\n", "Noted.\n")
    session.say("branch", split_names("train")[2][0])
    frames = session.take_activity()
    assert all(f["flagged"] == [] and f["depends"] == 0 for f in frames)
    session.say("card", CARD)
    session.take_activity()
    session.ask("name")
    assert any(f["flagged"] for f in session.take_activity())
    session.forget(flagged(session, "card").id)
    session.ask("card")
    frames = session.take_activity()
    assert all(f["flagged"] == [] and f["depends"] == 0 for f in frames)


def test_full_session_is_refused_before_reading(session):
    session.memory.position = 380
    with pytest.raises(SessionError) as error:
        session.ask("card")
    assert error.value.code == "session_full"
    assert session.memory.position == 380
