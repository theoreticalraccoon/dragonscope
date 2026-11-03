"""The public toolkit: capture, forget, report and dashboard."""

import json

import pytest
import torch
from fastapi.testclient import TestClient

import dragonscope as ds
from dragonscope.dashboard import MAX_POSITION, create_dashboard
from dragonscope.engine import SessionMemory
from dragonscope.model_config import demo_config
from dragonscope.vendor.bdh.bdh import BDH, BDHConfig

SECRET = "4111 1111 1111 1111"
TEXT = f"U: My card is {SECRET}.\nA: Noted.\nU: Hello there.\nA: Yes.\n"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return BDH(demo_config()).eval()


def test_capture_records_every_byte_and_matches_the_engine(model):
    memory = ds.capture(model).read(TEXT)
    plain = SessionMemory(model)
    plain.read(TEXT.encode())
    assert len(memory.history) == len(TEXT)
    assert torch.equal(memory.rho, plain.rho)
    assert memory.text == TEXT
    assert len(memory.history[0].firing) == model.config.n_layer
    assert memory.neuron_strength().shape == (4, 1024)


def test_capture_works_with_other_model_sizes():
    torch.manual_seed(1)
    small = BDH(BDHConfig(n_layer=2, n_embd=64, n_head=2, mlp_internal_dim_multiplier=8)).eval()
    memory = ds.capture(small).read("abc")
    assert memory.neuron_strength().shape == (2, 256)
    assert memory.forget("b").bytes_erased == 1


def test_forget_removes_every_copy_and_reports_a_residual(model):
    memory = ds.capture(model).read(f"{TEXT}{SECRET}.\nTail text.\n")
    result = memory.forget(SECRET)
    assert result.copies == 2
    assert result.bytes_erased == 2 * len(SECRET)
    assert SECRET not in memory.text
    assert memory.find(SECRET) == []
    assert result.residual > 0.001
    assert memory.memory.next_logits is None


def test_rebuild_method_matches_never_reading_it(model):
    memory = ds.capture(model).read(TEXT)
    result = memory.forget(SECRET, "rebuild")
    assert result.residual < 1e-5


def test_rebuild_clears_a_residual(model):
    memory = ds.capture(model).read(TEXT)
    memory.forget(SECRET)
    assert memory.residual()["total"] > 0.001
    memory.rebuild()
    # Stepping byte by byte checks the batched rebuild independently.
    assert memory.residual(stepped=True)["total"] < 1e-5
    assert memory.residual()["total"] < 1e-5


def test_forget_range_and_errors(model):
    memory = ds.capture(model).read("abcdef")
    assert memory.forget((1, 3)).bytes_erased == 2
    assert memory.text == "adef"
    with pytest.raises(ValueError):
        memory.forget("not here")
    with pytest.raises(ValueError):
        memory.forget("a", "shred")


def test_forget_leaves_no_copy_in_what_was_recorded(model):
    memory = ds.capture(model, keep_states=True).read(TEXT)
    start, end = memory.find(SECRET)[0]
    memory.forget(SECRET)
    positions = {step.position for step in memory.history}
    assert positions.isdisjoint(range(start, end))
    for step in memory.history:
        assert (step.state is not None) == (step.position < start)
    assert SECRET.encode() not in bytes(step.byte for step in memory.history)


def test_fork_leaves_the_session_alone(model):
    memory = ds.capture(model).read(TEXT)
    rho = memory.rho.clone()
    trial = memory.fork()
    trial.forget(SECRET)
    trial.read("More.")
    assert torch.equal(memory.rho, rho)
    assert SECRET in memory.text


def test_generate_and_probabilities(model):
    memory = ds.capture(model).read("abc")
    probabilities = memory.next_byte_probabilities()
    assert abs(float(probabilities.sum()) - 1) < 1e-4
    position = memory.position
    written = memory.generate(5, stop=None)
    assert memory.position == position + 5
    assert len(written) >= 1


def test_report_checks_its_own_probes_could_recall(model):
    # An untrained model recalls nothing, so forgetting must not be credited.
    case = ds.Case(TEXT, [SECRET], [ds.Probe("U: What?\nA: ", SECRET, "forgotten")])
    report = ds.forgetting_report(model, [case], max_reply=5)
    assert report.recall["before"]["forgotten"] == 0
    assert report.checks["Probes could recall the value before forgetting"] is False
    assert report.passed is False
    assert "FAIL" in report.to_markdown()
    assert report.residual["rebuild"] < 1e-5
    assert report.residual["subtract"] > 0.001


def test_report_needs_a_probe_about_the_forgotten_value(model):
    kept_only = ds.Case(TEXT, [SECRET], [ds.Probe("U: Hi?\nA: ", "Yes", "kept")])
    with pytest.raises(ValueError):
        ds.forgetting_report(model, [kept_only], max_reply=5)
    unknown = ds.Case(TEXT, [SECRET], [ds.Probe("U: What?\nA: ", SECRET, "both")])
    with pytest.raises(ValueError):
        ds.forgetting_report(model, [unknown], max_reply=5)


def test_report_on_the_trained_model():
    trained = ds.load_model("models/best_model.pt")
    card = "4242 4242 4242 4242"
    case = ds.Case(
        f"U: My card number is {card}.\nA: Noted.\nU: My branch is Kandy.\nA: Noted.\n",
        [card],
        [
            ds.Probe("U: What is my card number?\nA: ", card, "forgotten"),
            ds.Probe("U: What is my branch?\nA: ", "Kandy", "kept"),
        ],
    )
    report = ds.forgetting_report(trained, [case])
    assert report.passed
    assert report.recall["before"]["forgotten"] == 1
    assert report.recall["subtract"]["forgotten"] == 0
    assert report.recall["subtract"]["kept"] == 1
    assert report.to_dict()["passed"] is True
    json.dumps(report.to_dict())


def test_report_keeps_no_forgotten_text_from_before(model):
    case = ds.Case(TEXT, [SECRET], [ds.Probe("U: What?\nA: ", SECRET, "forgotten")])
    report = ds.forgetting_report(model, [case], max_reply=5)
    assert all("before" not in sample for sample in report.samples)


# Dashboard


def connect(client):
    return client.websocket_connect("/ws")


def until_state(socket):
    frames = []
    while True:
        message = socket.receive_json()
        if message["type"] == "frame":
            frames.append(message)
        else:
            return message, frames


@pytest.fixture
def client(model):
    return TestClient(create_dashboard(ds.capture(model), pace=0))


def test_dashboard_page_and_files(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Memory debugger" in page.text
    for path in ("/inspector.js", "/inspector.css", "/brain.js", "/vendor/three.min.js"):
        assert client.get(path).status_code == 200
    for path in ("/", "/inspector.js", "/inspector.css"):
        text = client.get(path).text
        assert "http://" not in text and "https://" not in text


def test_dashboard_reads_watches_and_forgets(client):
    with connect(client) as socket:
        hello = socket.receive_json()
        assert hello["model"]["neurons"] == 4096
        socket.send_json({"type": "read", "text": TEXT})
        state, frames = until_state(socket)
        assert len(frames) == len(TEXT)
        assert frames[0]["change"] == []
        assert state["runs"][0]["text"] == TEXT
        assert state["can_generate"]

        socket.send_json({"type": "watch", "text": SECRET})
        state, _ = until_state(socket)
        assert len(state["watched"]) == 1
        assert max(state["memory"]["flagged"]) > 0

        socket.send_json({"type": "read", "text": "Q"})
        state, frames = until_state(socket)
        assert len(frames[0]["change"]) == 4

        socket.send_json({"type": "forget", "text": SECRET, "method": "subtract"})
        raw = socket.receive_text()
        state = json.loads(raw)
        assert SECRET not in raw
        assert state["last_edit"]["bytes_erased"] == len(SECRET)
        assert state["last_edit"]["residual"] > 0.001
        assert state["watched"] == []
        assert not state["can_generate"]
        assert set(state["memory"]["flagged"]) == {0}

        socket.send_json({"type": "rebuild"})
        state, _ = until_state(socket)
        assert state["last_edit"]["residual"] < 1e-5

        socket.send_json({"type": "reset"})
        state, _ = until_state(socket)
        assert state["runs"] == []


def test_dashboard_errors_do_not_repeat_what_was_sent(client):
    with connect(client) as socket:
        socket.receive_json()
        for message in (
            {"type": "forget", "text": SECRET},
            {"type": "watch", "text": SECRET},
            {"type": "report", "forget": SECRET, "probes": []},
            {
                "type": "report",
                "forget": SECRET,
                "probes": [{"prompt": "U: Hi?\nA: ", "expect": "Yes", "about": "kept"}],
            },
            {"type": "generate", "count": 5},
            {"type": "nope"},
            {"type": "forget", "text": SECRET, "method": "shred"},
        ):
            socket.send_json(message)
            raw = socket.receive_text()
            assert json.loads(raw)["type"] == "error"
            assert SECRET not in raw
        socket.send_text("x" * 5000)
        assert socket.receive_json()["type"] == "error"
        socket.send_json({"type": "read", "text": "a" * (MAX_POSITION + 1)})
        assert socket.receive_json()["type"] == "error"


def test_dashboard_report_leaves_the_session_alone(client):
    with connect(client) as socket:
        socket.receive_json()
        socket.send_json({"type": "read", "text": TEXT})
        before, _ = until_state(socket)
        socket.send_json(
            {
                "type": "report",
                "forget": SECRET,
                "probes": [{"prompt": "U: What?\nA: ", "expect": SECRET, "about": "forgotten"}],
            }
        )
        state, _ = until_state(socket)
        assert state["runs"] == before["runs"]
        assert state["report"]["passed"] is False
        assert "samples" not in state["report"]


def test_dashboard_refuses_other_origins(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}):
            pass


# Ledger modes


def test_compact_ledger_is_smaller_and_nearly_exact(model):
    exact = ds.capture(model).read(TEXT)
    compact = ds.capture(model, ledger="compact").read(TEXT)
    assert torch.equal(exact.rho, compact.rho)
    assert compact.ledger_bytes * 2 <= exact.ledger_bytes
    exact.forget(SECRET, "subtract")
    result = compact.forget(SECRET, "subtract")
    assert result.method == "subtract"
    assert float((exact.rho - compact.rho).norm() / exact.rho.norm()) < 1e-3
    assert compact.memory.ledger.keys() == exact.memory.ledger.keys()


def test_no_ledger_falls_back_to_rebuild(model):
    memory = ds.capture(model, ledger="off").read(TEXT)
    assert memory.ledger_bytes == 0
    with pytest.raises(ValueError):
        memory.forget(SECRET, "subtract")
    result = memory.forget(SECRET)
    assert result.method == "rebuild"
    assert result.residual < 1e-5
    assert SECRET not in memory.text


def test_ledger_can_skip_text_that_will_never_be_forgotten(model):
    memory = ds.capture(model)
    memory.read("System: be helpful.\n", ledger=False)
    unrecorded = memory.position
    memory.read(TEXT)
    assert set(memory.memory.ledger) == set(range(unrecorded, memory.position))
    assert memory.forget(SECRET).method == "subtract"
    assert memory.forget("helpful").method == "rebuild"
    # A rebuild keeps entries only for text that had them.
    assert min(memory.memory.ledger) >= unrecorded
    assert memory.memory.recording


def test_capture_works_with_a_separate_copy_of_upstream_bdh():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "dragonscope" / "vendor" / "bdh" / "bdh.py"
    spec = importlib.util.spec_from_file_location("upstream_bdh", path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    torch.manual_seed(2)
    other = upstream.BDH(upstream.BDHConfig(n_layer=2, n_embd=64, n_head=2, mlp_internal_dim_multiplier=8)).eval()
    data = b"hello there"
    memory = ds.capture(other).read(data)
    full = other(torch.tensor([list(data)]))[0][0, -1]
    assert (memory.memory.next_logits - full).abs().max() < 1e-3
