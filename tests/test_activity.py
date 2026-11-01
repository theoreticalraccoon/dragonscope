"""The display must describe the real model, so check it against the model's tensors."""

import base64

import pytest
import torch
import torch.nn.functional as F

from dragonscope.activity import ACTIVATION_SCALE, ActivityRecorder, model_facts
from dragonscope.engine import SessionMemory
from dragonscope.model_config import demo_config
from dragonscope.train import load_model
from dragonscope.vendor.bdh.bdh import BDH


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return BDH(demo_config()).eval()


def decode(text):
    return torch.frombuffer(bytearray(base64.b64decode(text)), dtype=torch.uint8).float()


def record(model, text=b"U: What is my card number?\nA: "):
    memory = SessionMemory(model)
    recorder = ActivityRecorder(model, memory.n_layer)
    memory.observer = recorder
    frames = []
    for byte in text:
        memory.read(bytes([byte]))
        frames.append(recorder.take_frame())
    return frames


def test_facts_match_the_model(model):
    facts = model_facts(model)
    heads, _, per_head = model.encoder.shape
    assert facts["heads"] == heads
    assert facts["per_head"] == per_head
    assert facts["neurons"] == heads * per_head == model.decoder.shape[0]
    assert facts["rounds"] == model.config.n_layer
    assert facts["parameters"] == sum(p.numel() for p in model.parameters())


def test_layers_are_rounds_of_the_same_neurons(model):
    # One encoder serves every layer. If this changes, the display is wrong.
    assert model.encoder.ndim == 3
    assert not any("layer" in name or name[0].isdigit() for name, _ in model.named_parameters())


def test_frame_holds_every_neurons_real_first_round_activation(model):
    byte = ord("7")
    frame = record(model, bytes([byte]))[0]
    with torch.no_grad():
        x = model.ln(model.embed(torch.tensor([[byte]])).unsqueeze(1))
        fired = F.relu(x @ model.encoder)[0, :, 0, :].flatten()
    assert frame["counts"][0] == int((fired > 0).sum())
    sent = decode(frame["x"][0])
    assert len(sent) == model.decoder.shape[0]
    assert (sent - (fired / ACTIVATION_SCALE * 255).floor()).abs().max() <= 1


def test_links_are_positive_learned_connections_between_active_neurons(model):
    frames = record(model)
    checked = 0
    n = model.encoder.size(2)
    for frame in frames:
        for index, links in enumerate(frame["links"]):
            fired = decode(frame["x"][index + 1])
            for source, target in links:
                column = model.encoder[target // n, :, target % n].detach()
                assert float(model.decoder[source].detach() @ column) > 0
                assert fired[target] > 0
                checked += 1
    assert checked > 0


def test_neighbours_respond_alike_on_the_trained_model():
    model = load_model("models/best_model.pt")
    facts = model_facts(model)
    with torch.no_grad():
        characters = torch.tensor([10, *range(32, 127)])
        tuning = F.relu(model.ln(model.embed(characters)) @ model.encoder)[0].T  # N, chars
    unit = F.normalize(tuning, dim=1)
    n = facts["per_head"]
    place = torch.tensor([facts["u"][:n], facts["v"][:n]]).T
    distance = torch.cdist(place, place)
    distance.fill_diagonal_(1e9)
    nearest = distance.topk(5, largest=False).indices
    assert (unit[:, None] * unit[nearest]).sum(-1).mean() > (unit @ unit.T).mean() + 0.1


def test_flagged_data_changes_firing_when_recalled_but_not_otherwise():
    """Live, firing must differ from a memory without the card only where it matters."""
    from dragonscope.session import DemoSession

    session = DemoSession(load_model("models/best_model.pt"))
    frames = []
    session.emit = frames.append
    session.say("card", "4532 0151 1283 0366")
    session.say("branch", "Kandy")

    def final_round_change(field):
        frames.clear()
        session.ask(field)
        writes = [f for f in frames if f["phase"] == "write"]
        digits = [f for f in writes if f["char"].isdigit()]
        return writes, digits

    writes, digits = final_round_change("card")
    assert len(digits) == 16
    assert sum(f["depends"] for f in digits) / len(digits) > 0.5
    card_change = sum(float(decode(f["change"][-1]).sum() / decode(f["x"][-1]).sum()) for f in digits) / len(digits)

    writes, _ = final_round_change("branch")
    assert all(f["depends"] < 0.05 for f in writes)
    branch_change = sum(float(decode(f["change"][-1]).sum() / decode(f["x"][-1]).sum()) for f in writes) / len(writes)

    assert card_change > 3 * branch_change
    # The first round depends only on the byte, so memory cannot change it.
    assert all(decode(f["change"][0]).sum() == 0 for f in frames)
