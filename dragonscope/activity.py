"""What the display shows about the real model, taken from its own tensors.

The model has ``n_head * N`` neurons. Its encoder is shared by every layer, so a
byte passes through the same neurons once per layer. These are called rounds here,
as they are not separate sets of neurons.

A frame is sent live for each byte as the model computes it. For every round it
holds every neuron's activation, how many neurons fired, how many passed a signal
on, and the learned connections most used. When something is flagged, a second copy
of memory that never read the flagged bytes runs alongside. ``change`` is how much
each neuron's firing differs from that copy, which is what the flagged data caused.

Frames are sensitive: they hold the byte itself. Nothing on the server keeps them,
and the page drops them when the next ``state`` arrives.
"""

from __future__ import annotations

import base64

import torch
import torch.nn.functional as F

from dragonscope.vendor.bdh.bdh import BDH

# Activations are sent as 0 to 255 on this scale. The largest measured was 6.96.
ACTIVATION_SCALE = 7.0
TOP_LINKS = 4
# A trained model passes about 100 signals a round. This bounds the search if not.
MAX_LINK_SOURCES = 256


def encode_activation(values: torch.Tensor) -> str:
    """Pack one value per neuron into base64, with one byte for each."""
    scaled = (values.flatten() / ACTIVATION_SCALE * 255).clamp(0, 255)
    return base64.b64encode(scaled.to(torch.uint8).cpu().numpy().tobytes()).decode("ascii")


class ActivityRecorder:
    """Watches ``SessionMemory`` through its observer, building one frame per byte."""

    def __init__(self, model: BDH, n_layer: int):
        self.model = model
        self.n_layer = n_layer
        self._counterfactual: dict[int, torch.Tensor] = {}
        self._frame: dict = {}
        self._previous_passed: torch.Tensor | None = None

    def watch_counterfactual(self, level: int, x_sparse, xy_sparse, qr) -> None:
        """Observer for the copy of memory that never read the flagged bytes."""
        self._counterfactual[level] = x_sparse.clone()

    def _links(self, passed: torch.Tensor, fired: torch.Tensor) -> list[list[int]]:
        """Find the connections that most carried one round into the next.

        A neuron's output reaches the next round through its row of the decoder, and
        is read by the next neurons through their encoder columns. So the learned
        connection from i to j is ``decoder[i] . encoder[j]``. It counts here in
        proportion to the signal i passed and how strongly j then fired. This leaves
        out the normalisation between rounds.
        """
        sources = passed.nonzero().flatten()
        targets = fired.nonzero().flatten()
        if not len(sources) or not len(targets):
            return []
        if len(sources) > MAX_LINK_SOURCES:
            sources = passed.topk(MAX_LINK_SOURCES).indices
        n = self.model.encoder.size(2)
        columns = self.model.encoder[targets // n, :, targets % n]
        weights = self.model.decoder[sources] @ columns.T
        scores = weights * passed[sources][:, None] * fired[targets][None, :]
        best = scores.flatten().topk(min(TOP_LINKS, scores.numel()))
        width = len(targets)
        return [
            [int(sources[place // width]), int(targets[place % width])]
            for place, score in zip(best.indices.tolist(), best.values.tolist())
            if score > 0
        ]

    @torch.no_grad()
    def __call__(self, level: int, x_sparse, xy_sparse, qr) -> None:
        if level == 0:
            self._frame = {"x": [], "counts": [], "passed": [], "links": [], "change": []}
            self._previous_passed = None
        frame = self._frame
        fired = x_sparse.flatten()
        passed = xy_sparse.flatten()
        frame["x"].append(encode_activation(fired))
        frame["counts"].append(int((fired > 0).sum()))
        frame["passed"].append(int((passed > 0).sum()))
        if self._previous_passed is not None:
            frame["links"].append(self._links(self._previous_passed, fired))
        self._previous_passed = passed
        if level in self._counterfactual:
            frame["change"].append(
                encode_activation((x_sparse - self._counterfactual[level]).abs())
            )

    def take_frame(self) -> dict:
        frame, self._frame = self._frame, {}
        self._counterfactual = {}
        return frame


@torch.no_grad()
def model_facts(model: BDH) -> dict:
    """Describe the model, and place each neuron by what it responds to.

    Each neuron's firing for every text character is its tuning. The two main axes
    of those tunings, within a head, give its ``u`` and ``v``, so neighbours respond
    to similar characters. Ranks spread the neurons evenly from 0 to 1.
    """
    config = model.config
    n_head = config.n_head
    n = config.mlp_internal_dim_multiplier * config.n_embd // n_head
    characters = torch.tensor([10, *range(32, 127)], device=model.encoder.device)
    tuning = F.relu(model.ln(model.embed(characters)) @ model.encoder)  # h, chars, N

    us, vs = [], []
    for head in range(n_head):
        centred = tuning[head].T - tuning[head].T.mean(dim=0, keepdim=True)
        axes = torch.linalg.svd(centred, full_matrices=False).U[:, :2]
        ranks = axes.argsort(dim=0).argsort(dim=0).float() / (n - 1)
        us += [round(value, 4) for value in ranks[:, 0].tolist()]
        vs += [round(value, 4) for value in ranks[:, 1].tolist()]
    return {
        "neurons": n_head * n,
        "heads": n_head,
        "per_head": n,
        "rounds": config.n_layer,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "u": us,
        "v": vs,
    }
