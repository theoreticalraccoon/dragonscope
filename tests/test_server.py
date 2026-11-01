import json

import pytest
import torch
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from dragonscope.model_config import demo_config
from dragonscope.server import create_app
from dragonscope.vendor.bdh.bdh import BDH

CARD = "4111 1111 1111 1111"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return BDH(demo_config()).eval()


@pytest.fixture
def client(model):
    return TestClient(create_app(model))


def next_state(socket):
    """Skip the activity frames that arrive before a chat turn's state."""
    message = socket.receive_json()
    if message["type"] == "activity":
        assert message["frames"]
        message = socket.receive_json()
    assert message["type"] == "state"
    return message


def test_hello_has_the_guide_and_state(client):
    with client.websocket_connect("/ws") as socket:
        hello = socket.receive_json()
    assert hello["type"] == "hello"
    assert "statements" in hello["guide"]
    assert hello["state"]["turns"] == []


def test_say_forget_and_reset_over_the_socket(client):
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        socket.send_json({"type": "say", "field": "card", "value": CARD})
        state = next_state(socket)
        finding = next(f for f in state["findings"] if f["source"] == "user")

        socket.send_json({"type": "forget", "finding": finding["id"]})
        raw_result = socket.receive_text()
        raw_state = socket.receive_text()
        result = json.loads(raw_result)
        assert result["type"] == "forget_result"
        assert result["log_entry"]["last4"] == "1111"
        assert CARD not in raw_result and CARD not in raw_state
        assert json.loads(raw_state)["deletion_log"] == [result["log_entry"]]

        socket.send_json({"type": "reset"})
        assert socket.receive_json()["turns"] == []


def test_every_client_gets_the_new_state(client):
    with client.websocket_connect("/ws") as first, client.websocket_connect("/ws") as second:
        first.receive_json()
        second.receive_json()
        first.send_json({"type": "say", "field": "card", "value": CARD})
        assert next_state(first)["findings"]
        assert next_state(second)["findings"]


def test_errors_go_back_without_repeating_the_value(client):
    wrong = "4111 1111 1111 1112"
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        socket.send_json({"type": "say", "field": "card", "value": wrong})
        raw = socket.receive_text()
        assert json.loads(raw)["code"] == "invalid_value"
        assert wrong not in raw

        for bad, code in (
            ("not json", "bad_json"),
            ("[1]", "bad_json"),
            ('{"type": "nope"}', "unknown_type"),
            ('{"type": "ask", "field": "pin"}', "unknown_field"),
            ('{"type": "forget", "finding": 99}', "unknown_finding"),
            ('{"type": "say", "field": "card"}', "invalid_value"),
            ("x" * 5000, "too_long"),
        ):
            socket.send_text(bad)
            assert socket.receive_json()["code"] == code


def test_other_origins_are_refused(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}):
            pass


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_page_and_its_files_are_served_locally(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "DragonScope" in page.text
    for path in ("style.css", "app.js", "brain.js", "vendor/three.min.js"):
        assert client.get(f"/{path}").status_code == 200
    # The demo must run offline, so our own files load nothing from elsewhere.
    for path in ("/", "/style.css", "/app.js", "/brain.js"):
        assert "http://" not in client.get(path).text
        assert "https://" not in client.get(path).text
