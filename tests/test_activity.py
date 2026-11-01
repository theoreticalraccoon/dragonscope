"""The display must describe the real model, so check it against the model's tensors."""

import pytest
import torch
import torch.nn.functional as F

from dragonscope.activity import ActivityRecorder, model_facts
from dragonscope.engine import SessionMemory
from dragonscope.model_config import demo_config
from dragonscope.train import load_model
from dragonscope.vendor.bdh.bdh import BDH


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return BDH(demo_config()).eval()


def record(model, text=b"U: What is my card number?\nA: "):
    memory = SessionMemory(model)
    recorder = ActivityRecorder(model, memory.n_layer, None)
    memory.observer = recorder
    memory.read(text)
    return recorder.frames


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


def test_counts_and_firing_match_the_first_rounds_real_activation(model):
    byte = ord("7")
    frames = record(model, bytes([byte]))
    with torch.no_grad():
        x = model.ln(model.embed(torch.tensor([[byte]])).unsqueeze(1))
        fired = F.relu(x @ model.encoder)[0, :, 0, :].flatten()
    assert frames[0]["counts"][0] == int((fired > 0).sum())
    assert frames[0]["fired"][0]["i"] == fired.topk(300).indices[: len(frames[0]["fired"][0]["i"])].tolist()


def test_links_are_positive_learned_connections_between_firing_neurons(model):
    frames = record(model)
    checked = 0
    for frame in frames:
        for round_index, links in enumerate(frame["links"]):
            fired = set(frame["fired"][round_index + 1]["i"])
            for source, target in links:
                column = model.encoder[target // model.encoder.size(2), :, target % model.encoder.size(2)]
                assert float(model.decoder[source] @ column) > 0
                assert target in fired
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
