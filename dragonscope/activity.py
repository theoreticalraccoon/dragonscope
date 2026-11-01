"""Record what the model really does while it reads or writes each byte.

Each frame holds the neurons that fired at each layer, and the neurons where this
byte's query meets flagged memory. These come from the tensors that compute the
reply. ``depends`` is set afterwards by comparing the reply with a counterfactual
run in which the flagged bytes were never read: it is how much of the probability
of writing this byte disappears without them.

Frames hold no text. They are still sensitive: at layer 0, which neurons fire
depends only on the byte, so a frame could be mapped back to it. They are sent
once and never stored by the server, and the page drops them after playing them.
"""

from __future__ import annotations

import torch

TOP_FIRED = 24  # Neurons reported per layer, per byte.
TOP_FLAGGED = 16


def _top(values: torch.Tensor, count: int, offset: int) -> dict[str, list]:
    """Return the strongest entries, with strengths scaled so the top one is 1."""
    strengths, indices = values.flatten().topk(count)
    peak = float(strengths[0])
    if peak <= 0:
        return {"i": [], "s": []}
    keep = strengths > 0
    return {
        "i": (indices[keep] + offset).tolist(),
        "s": [round(value, 2) for value in (strengths[keep] / peak).tolist()],
    }


class ActivityRecorder:
    """Watches ``SessionMemory`` through its observer, one frame per byte."""

    def __init__(
        self, n_head: int, n: int, n_layer: int, flagged_rho: torch.Tensor | None
    ):
        self.per_layer = n_head * n
        self.n_layer = n_layer
        self.flagged_rho = flagged_rho
        self.phase = "read"
        self.frames: list[dict] = []
        self._current: dict = {}

    @torch.no_grad()
    def __call__(self, level: int, x_sparse, qr, rho) -> None:
        if level == 0:
            self._current = {
                "phase": self.phase,
                "fired": [],
                "flagged": [],
                "depends": 0.0,
            }
        frame = self._current
        offset = level * self.per_layer
        frame["fired"].append(_top(x_sparse, TOP_FIRED, offset))
        if self.flagged_rho is not None:
            frame["flagged"].append(
                _top(self.flagged_rho[level].norm(dim=1) * qr.abs(), TOP_FLAGGED, offset)
            )
        if level == self.n_layer - 1:
            self.frames.append(frame)
