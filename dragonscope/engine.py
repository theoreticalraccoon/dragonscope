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

Ledger modes, set per session, trade memory for how ``subtract`` works:

* ``exact`` keeps each byte's entry in full precision. This is the default.
* ``compact`` stores entries in float16, keeping only the non-zero part of
  ``QR`` when that is smaller. It is at least 2 times smaller, and 2.7 times for
  the demo model. Subtracting it adds a relative error of about 1e-4 to ``rho``.
* ``off`` keeps no entries. Forgetting such bytes needs ``replay``.

``recording`` can also be switched off for parts of a session, such as text
that will never need forgetting, which saves the most memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dragonscope.vendor.bdh.bdh import BDH, Attention


LEDGER_MODES = ("exact", "compact", "off")
# Bytes subtracted together, which bounds the temporary memory used.
SUBTRACT_CHUNK = 128


@dataclass
class LedgerEntry:
    """One byte's factorised contribution to ``rho``.

    Layer 0 of ``v`` maps straight back to the byte, so an entry is as sensitive
    as the plaintext and must be destroyed by Forget.
    """

    v: torch.Tensor  # (n_layer, D)
    qr: torch.Tensor  # (n_layer, n_head, N)

    @property
    def nbytes(self) -> int:
        return self.v.numel() * self.v.element_size() + self.qr.numel() * self.qr.element_size()


class CompactLedgerEntry:
    """A ``LedgerEntry`` stored in float16, keeping only the non-zero part of ``qr``
    when that is smaller.

    Sparse storage costs an index per value, so a byte that fires most neurons is
    kept dense instead. Either way ``qr`` takes at most half the space of an exact
    entry. ``v`` and ``qr`` read back as full tensors in the original dtype.
    """

    __slots__ = ("_v", "_index", "_values", "_shape", "_dtype")

    def __init__(self, v: torch.Tensor, qr: torch.Tensor):
        flat = qr.reshape(-1)
        index_dtype = torch.int16 if flat.numel() <= 32768 else torch.int32
        index = flat.nonzero().squeeze(1)
        sparse_bytes = index.numel() * (index_dtype.itemsize + 2)
        if sparse_bytes < flat.numel() * 2:
            self._index = index.to(index_dtype)
            self._values = flat[index].to(torch.float16)
        else:
            self._index = None
            self._values = flat.to(torch.float16)
        self._v = v.to(torch.float16)
        self._shape = tuple(qr.shape)
        self._dtype = qr.dtype

    @property
    def v(self) -> torch.Tensor:
        return self._v.to(self._dtype)

    @property
    def qr(self) -> torch.Tensor:
        if self._index is None:
            return self._values.to(self._dtype).view(self._shape)
        size = 1
        for dimension in self._shape:
            size *= dimension
        flat = torch.zeros(size, dtype=self._dtype, device=self._values.device)
        flat[self._index.long()] = self._values.to(self._dtype)
        return flat.view(self._shape)

    @property
    def nbytes(self) -> int:
        tensors = [self._v, self._values] + ([] if self._index is None else [self._index])
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


class SessionMemory:
    """Reads bytes one at a time, keeping ``rho`` and a per-position ledger."""

    def __init__(self, model: BDH, ledger: str = "exact"):
        if ledger not in LEDGER_MODES:
            raise ValueError(f"unknown ledger mode {ledger!r}: use one of {LEDGER_MODES}")
        self.model = model.eval()
        self.ledger_mode = ledger
        # Set to False to read bytes without ledger entries. They can then only be
        # forgotten with ``replay``.
        self.recording = True
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
        other = SessionMemory(self.model, self.ledger_mode)
        other.recording = self.recording
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
        if self.recording and self.ledger_mode != "off":
            self.ledger[position] = self._entry(torch.stack(entry_v), torch.stack(entry_qr))
        return x.view(self.D) @ model.lm_head

    def _entry(self, v: torch.Tensor, qr: torch.Tensor):
        if self.ledger_mode == "compact":
            return CompactLedgerEntry(v, qr)
        return LedgerEntry(v=v, qr=qr)

    def ledger_bytes(self) -> int:
        """Return the memory held by ledger entries, in bytes."""
        return sum(entry.nbytes for entry in self.ledger.values())

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

    def forget(
        self, start: int, end: int, method: str = "subtract", defer_rebuild: bool = False
    ) -> list[int]:
        """Erase positions ``start <= p < end`` and return those actually erased.

        With ``replay`` and ``defer_rebuild``, the bytes are dropped but ``rho`` still
        holds them until the caller runs ``rebuild()``. Use it to erase several spans
        with one rebuild.
        """
        if method not in ("subtract", "replay"):
            raise ValueError(f"unknown forget method: {method}")
        positions = [p for p in range(start, end) if p in self.bytes]

        if method == "subtract":
            missing = [p for p in positions if p not in self.ledger]
            if missing:
                raise ValueError(
                    f"{len(missing)} of these bytes have no ledger entry, so use replay"
                )
            # One pass over rho per chunk of bytes, rather than one per byte.
            for first in range(0, len(positions), SUBTRACT_CHUNK):
                entries = [self.ledger[p] for p in positions[first : first + SUBTRACT_CHUNK]]
                v = torch.stack([entry.v for entry in entries], dim=1)  # L, T, D
                qr = torch.stack([entry.qr for entry in entries], dim=2)  # L, H, T, N
                for level in range(self.n_layer):
                    self.rho[level] -= torch.einsum("td,htn->hdn", v[level], qr[level])

        for position in positions:
            del self.bytes[position]
            self.ledger.pop(position, None)
        # These logits were computed while the secret was still in memory.
        self.next_logits = None

        if method == "replay" and not defer_rebuild:
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
        recorded = set(self.ledger)
        position = self.position
        self.reset()
        self.position = position
        if not survivors:
            return

        _, rho, v, qr = self._forward_all(survivors)
        self.rho = rho
        for slot, (original_position, byte) in enumerate(survivors):
            self.bytes[original_position] = byte
            if original_position in recorded:
                self.ledger[original_position] = self._entry(
                    v[:, slot].clone(), qr[:, :, slot].clone()
                )

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
        clean = SessionMemory(self.model, "off")
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
