"""Stateful BDH inference with an inspectable, erasable session memory.

The vendored ``BDH.forward`` recomputes attention from the whole context on
every call and keeps nothing between calls. Its attention term can be rewritten
as a running matrix, which is the memory this project shows and erases:

    scores @ V  at position t  =  sum_{tau<t} (QR_t . QR_tau) V_tau
                               =  rho @ QR_t
    where  rho  =  sum_{tau<t} V_tau QR_tau^T        (one per layer and head)

``rho`` has shape (n_layer, n_head, D, N). The paper describes this as a
compressed form of the BDH synaptic state.

Forget methods:

* ``subtract`` removes each forgotten byte's own ledger entries from ``rho``.
  Layer 0 becomes exact. Later bytes absorbed some of the secret at deeper
  layers, so layers >= 1 can keep a residual; ``residual_from_replay`` measures it.
* ``replay`` rebuilds ``rho`` from the surviving bytes at their original
  positions, which matches a session that never saw the forgotten bytes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dragonscope.vendor.bdh.bdh import BDH, Attention


@dataclass
class LedgerEntry:
    """One byte's factorised contribution to ``rho``.

    Layer 0 of ``v`` maps straight back to the byte, so an entry is as sensitive
    as the plaintext and must be destroyed by Forget.
    """

    v: torch.Tensor  # (n_layer, D)
    qr: torch.Tensor  # (n_layer, n_head, N)


class SessionMemory:
    """Reads bytes one at a time, keeping ``rho`` and a per-position ledger."""

    def __init__(self, model: BDH):
        self.model = model.eval()
        config = model.config
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.D = config.n_embd
        self.N = config.mlp_internal_dim_multiplier * config.n_embd // config.n_head
        # Optional callback(level, x_sparse, xy_sparse, qr) to watch a step's layers.
        # It must only read. ``fork`` and ``rebuild`` do not report to it.
        self.observer = None
        self.reset()

    def reset(self) -> None:
        parameter = next(self.model.parameters())
        self.rho = torch.zeros(
            self.n_layer,
            self.n_head,
            self.D,
            self.N,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        self.position = 0
        # Surviving bytes and ledger entries, both keyed by original position.
        self.bytes: dict[int, int] = {}
        self.ledger: dict[int, LedgerEntry] = {}
        self.next_logits: torch.Tensor | None = None

    def fork(self) -> "SessionMemory":
        """Return an independent copy, so one history can try several continuations."""
        other = SessionMemory(self.model)
        other.rho = self.rho.clone()
        other.position = self.position
        other.bytes = dict(self.bytes)
        other.ledger = dict(self.ledger)  # Entries are never changed in place.
        other.next_logits = (
            None if self.next_logits is None else self.next_logits.clone()
        )
        return other

    def text(self) -> bytes:
        """Return surviving bytes in order; forgotten positions are simply absent."""
        return bytes(byte for _, byte in sorted(self.bytes.items()))

    def find_positions(self, value: bytes) -> list[tuple[int, int]]:
        """Return the ``(start, end)`` of each contiguous occurrence of ``value``."""
        spans = []
        for start in range(self.position - len(value) + 1):
            if all(
                self.bytes.get(start + offset) == byte
                for offset, byte in enumerate(value)
            ):
                spans.append((start, start + len(value)))
        return spans

    @torch.no_grad()
    def _step(self, byte: int, position: int) -> torch.Tensor:
        """Mirror one position of ``BDH.forward`` using ``rho`` as attention."""
        model = self.model
        device = self.rho.device
        index = torch.tensor([[byte]], device=device)
        x = model.ln(model.embed(index).unsqueeze(1))  # 1, 1, 1, D
        phases = position * model.attn.freqs
        entry_v = []
        entry_qr = []

        for level in range(self.n_layer):
            x_sparse = F.relu(x @ model.encoder)  # 1, nh, 1, N
            qr = Attention.rope(phases, x_sparse)[0, :, 0, :]  # nh, N
            v = x[0, 0, 0, :]  # D

            attention = torch.einsum("hdn,hn->hd", self.rho[level], qr)
            self.rho[level] += torch.einsum("d,hn->hdn", v, qr)
            entry_v.append(v)
            entry_qr.append(qr)

            y_kv = model.ln(attention.view(1, self.n_head, 1, self.D))
            y_sparse = F.relu(y_kv @ model.encoder_v)
            xy_sparse = x_sparse * y_sparse
            if self.observer is not None:
                self.observer(level, x_sparse[0, :, 0, :], xy_sparse[0, :, 0, :], qr)
            y_mlp = (
                xy_sparse.transpose(1, 2).reshape(1, 1, 1, self.N * self.n_head)
                @ model.decoder
            )
            x = model.ln(x + model.ln(y_mlp))

        self.bytes[position] = byte
        self.ledger[position] = LedgerEntry(
            v=torch.stack(entry_v), qr=torch.stack(entry_qr)
        )
        return x.view(self.D) @ model.lm_head

    def read(self, data: bytes) -> torch.Tensor:
        """Read bytes into memory and return one row of next-byte logits per byte."""
        if not data:
            raise ValueError("read() needs at least one byte")
        rows = []
        for byte in data:
            rows.append(self._step(byte, self.position))
            self.position += 1
        self.next_logits = rows[-1]
        return torch.stack(rows)

    def generate(self, max_new_bytes: int, stop_byte: int | None = 10) -> bytes:
        """Greedily extend the session. The bytes written also enter memory."""
        if self.next_logits is None:
            raise RuntimeError("read a prompt first: Forget clears the pending logits")
        output = bytearray()
        for _ in range(max_new_bytes):
            byte = int(self.next_logits.argmax())
            output.append(byte)
            self.read(bytes([byte]))
            if byte == stop_byte:
                break
        return bytes(output)

    def forget(self, start: int, end: int, method: str = "subtract") -> list[int]:
        """Erase positions ``start <= p < end`` and return those actually erased."""
        if method not in ("subtract", "replay"):
            raise ValueError(f"unknown forget method: {method}")
        positions = [p for p in range(start, end) if p in self.bytes]

        if method == "subtract":
            for position in positions:
                entry = self.ledger[position]
                for level in range(self.n_layer):
                    self.rho[level] -= torch.einsum(
                        "d,hn->hdn", entry.v[level], entry.qr[level]
                    )

        for position in positions:
            del self.bytes[position]
            del self.ledger[position]
        # These logits were computed while the secret was still in memory.
        self.next_logits = None

        if method == "replay":
            self.rebuild()
        return positions

    @torch.no_grad()
    def rebuild(self) -> None:
        """Recompute ``rho`` and the ledger from surviving bytes at their positions.

        This runs each layer over all bytes at once, like ``BDH.forward`` but with
        the original positions, which is many times faster than stepping byte by
        byte. ``residual_from_replay`` still steps, so it checks this independently.
        """
        survivors = sorted(self.bytes.items())
        position = self.position
        self.reset()
        self.position = position
        if not survivors:
            return

        _, rho, v, qr = self._forward_all(survivors)
        self.rho = rho
        for slot, (original_position, byte) in enumerate(survivors):
            self.bytes[original_position] = byte
            self.ledger[original_position] = LedgerEntry(
                v=v[:, slot].clone(), qr=qr[:, :, slot].clone()
            )

    @torch.no_grad()
    def logits_without(self, excluded: set[int]) -> dict[int, torch.Tensor]:
        """Return next-byte logits per position, as if ``excluded`` were never read.

        This changes nothing in the session. Comparing it with ``excluded`` empty
        shows what the model's output owes to those bytes.
        """
        survivors = [item for item in sorted(self.bytes.items()) if item[0] not in excluded]
        if not survivors:
            return {}
        logits, *_ = self._forward_all(survivors)
        return {position: logits[slot] for slot, (position, _) in enumerate(survivors)}

    def _forward_all(self, survivors: list[tuple[int, int]]):
        """Run all ``(position, byte)`` pairs at once and return logits, rho, v and qr."""
        model = self.model
        device = self.rho.device
        count = len(survivors)
        positions = torch.tensor(
            [p for p, _ in survivors], device=device, dtype=model.attn.freqs.dtype
        ).view(1, 1, -1, 1)
        index = torch.tensor([[byte for _, byte in survivors]], device=device)
        x = model.ln(model.embed(index).unsqueeze(1))  # 1, 1, T, D
        phases = positions * model.attn.freqs
        rho = torch.zeros_like(self.rho)
        all_v = []
        all_qr = []

        for level in range(self.n_layer):
            x_sparse = F.relu(x @ model.encoder)  # 1, nh, T, N
            qr = Attention.rope(phases, x_sparse)
            attention = (qr @ qr.mT).tril(diagonal=-1) @ x
            rho[level] = x[0, 0].T @ qr[0]
            all_v.append(x[0, 0])
            all_qr.append(qr[0])

            y_sparse = F.relu(model.ln(attention) @ model.encoder_v)
            y_mlp = (x_sparse * y_sparse).transpose(1, 2).reshape(
                1, 1, count, self.N * self.n_head
            ) @ model.decoder
            x = model.ln(x + model.ln(y_mlp))

        logits = x[0, 0] @ model.lm_head  # T, vocab
        # n_layer, T, D and n_layer, nh, T, N
        return logits, rho, torch.stack(all_v), torch.stack(all_qr)

    def residual_from_replay(self) -> dict[str, list[float] | float]:
        """Compare ``rho`` with a clean rebuild of the surviving bytes.

        ``total`` is the relative Frobenius distance, and ``per_layer`` splits it
        by layer. A session that used ``replay`` reports zero up to float error.
        """
        clean = SessionMemory(self.model)
        for position, byte in sorted(self.bytes.items()):
            clean._step(byte, position)
        difference = self.rho - clean.rho
        per_layer = []
        for level in range(self.n_layer):
            norm = clean.rho[level].norm().clamp_min(1e-12)
            per_layer.append(float(difference[level].norm() / norm))
        total = float(difference.norm() / clean.rho.norm().clamp_min(1e-12))
        return {"total": total, "per_layer": per_layer}

    def ledger_sum(self) -> torch.Tensor:
        """Sum all ledger entries; this equals ``rho`` until a Forget changes history."""
        total = torch.zeros_like(self.rho)
        for entry in self.ledger.values():
            for level in range(self.n_layer):
                total[level] += torch.einsum(
                    "d,hn->hdn", entry.v[level], entry.qr[level]
                )
        return total
