"""What the display shows about the real model, taken from its own tensors.

The model has ``n_head * N`` neurons. Its encoder is shared by every layer, so a
byte passes through the same neurons once per layer. These are called rounds here,
as they are not separate sets of neurons.

Each frame covers one byte. It holds, per round, the neurons that fired and the
learned connections most used between them, plus the neurons where the byte's query
meets flagged memory. ``depends`` is set afterwards by comparing the reply with a
counterfactual run in which the flagged bytes were never read: it is how much of
the probability of writing this byte disappears without them.

Frames hold no text. They are still sensitive: in the first round, which neurons
fire depends only on the byte, so a frame could be mapped back to it. They are sent
once and never stored by the server, and the page drops them after playing them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dragonscope.vendor.bdh.bdh import BDH

TOP_FIRED = 300  # Strongest firing neurons reported per round. About 880 fire.
TOP_FLAGGED = 48
LINK_CANDIDATES = 48  # Strongest neurons considered at each end of a connection.
TOP_LINKS = 8


def _top(values: torch.Tensor, count: int) -> dict[str, list]:
    """Return the strongest entries, with strengths scaled so the top one is 1."""
    strengths, indices = values.flatten().topk(count)
    peak = float(strengths[0])
    if peak <= 0:
        return {"i": [], "s": []}
    keep = strengths > 0
    return {
        "i": indices[keep].tolist(),
        "s": [round(value, 2) for value in (strengths[keep] / peak).tolist()],
    }


class ActivityRecorder:
    """Watches ``SessionMemory`` through its observer, one frame per byte."""

    def __init__(self, model: BDH, n_layer: int, flagged_rho: torch.Tensor | None):
        self.model = model
        self.n_layer = n_layer
        self.flagged_rho = flagged_rho
        self.phase = "read"
        self.frames: list[dict] = []
        self._current: dict = {}
        self._previous_passed: torch.Tensor | None = None

    def _links(self, passed: torch.Tensor, fired: torch.Tensor) -> list[list[int]]:
        """Find the connections that most carried one round into the next.

        A neuron's output reaches the next round through its row of the decoder, and
        is read by the next neurons through their encoder columns. So the learned
        connection from i to j is ``decoder[i] . encoder[j]``, and it counts here
        when i passed a signal and j then fired.
        """
        n = self.model.encoder.size(2)
        sources = passed.topk(LINK_CANDIDATES)
        targets = fired.topk(LINK_CANDIDATES)
        columns = self.model.encoder[targets.indices // n, :, targets.indices % n]
        weights = self.model.decoder[sources.indices] @ columns.T
        scores = weights * sources.values[:, None] * targets.values[None, :]
        best = scores.flatten().topk(TOP_LINKS)
        links = []
        for position, score in zip(best.indices.tolist(), best.values.tolist()):
            if score <= 0:
                break
            links.append(
                [
                    int(sources.indices[position // LINK_CANDIDATES]),
                    int(targets.indices[position % LINK_CANDIDATES]),
                ]
            )
        return links

    @torch.no_grad()
    def __call__(self, level: int, x_sparse, xy_sparse, qr) -> None:
        if level == 0:
            self._current = {
                "phase": self.phase,
                "fired": [],
                "counts": [],
                "links": [],
                "flagged": [],
                "depends": 0.0,
            }
            self._previous_passed = None
        frame = self._current
        fired = x_sparse.flatten()
        frame["fired"].append(_top(fired, TOP_FIRED))
        frame["counts"].append(int((fired > 0).sum()))
        if self._previous_passed is not None:
            frame["links"].append(self._links(self._previous_passed, fired))
        self._previous_passed = xy_sparse.flatten()
        if self.flagged_rho is not None:
            frame["flagged"].append(
                _top(self.flagged_rho[level].norm(dim=1) * qr.abs(), TOP_FLAGGED)
            )
        if level == self.n_layer - 1:
            self.frames.append(frame)


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
