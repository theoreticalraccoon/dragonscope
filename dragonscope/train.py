"""Train the demo BDH. Safe to stop and rerun: it resumes from ``latest.pt``.

Run: ``python -m dragonscope.train --data data --out runs/main``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dragonscope.data import FIELDS, KIND_RECALL, KIND_REFUSAL
from dragonscope.model_config import demo_config
from dragonscope.vendor.bdh.bdh import BDH, BDHConfig

PAD_TARGET = -100


class Conversations:
    def __init__(self, path: Path):
        with np.load(path) as archive:
            self.bytes = archive["bytes"]
            self.offsets = archive["offsets"]
            self.answers = archive["answers"]
        self.count = len(self.offsets) - 1

    def batch(self, indices, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return inputs and targets, padded at the end of each conversation."""
        lengths = [int(self.offsets[i + 1] - self.offsets[i]) for i in indices]
        width = max(lengths)
        inputs = np.zeros((len(indices), width - 1), dtype=np.int64)
        targets = np.full((len(indices), width - 1), PAD_TARGET, dtype=np.int64)
        for row, (index, length) in enumerate(zip(indices, lengths)):
            conversation = self.bytes[self.offsets[index] : self.offsets[index] + length]
            inputs[row, : length - 1] = conversation[:-1]
            targets[row, : length - 1] = conversation[1:]
        return (
            torch.from_numpy(inputs).to(device),
            torch.from_numpy(targets).to(device),
        )


def learning_rate(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))


def atomic_save(payload: dict, path: Path) -> None:
    """Write then rename, retrying because Windows and OneDrive can hold files open."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(1 + attempt)


def load_model(path: Path | str, device: str = "cpu") -> BDH:
    """Load weights from either ``latest.pt`` or ``best_model.pt``, for inference."""
    payload = torch.load(path, map_location=device, weights_only=False)
    model = BDH(BDHConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


@torch.no_grad()
def evaluate(model: BDH, data: Conversations, limit: int, batch_size: int, device) -> dict:
    """Score whole replies by exact match.

    Teacher forcing gives the same verdict as greedy generation here: a reply is
    an exact match only if every next-byte prediction is right, and in that case
    greedy decoding would have followed the same bytes.
    """
    model.eval()
    limit = min(limit, data.count)
    correct_by_conversation: dict[int, list[tuple[int, int, bool]]] = {}
    losses = []
    for begin in range(0, limit, batch_size):
        indices = list(range(begin, min(begin + batch_size, limit)))
        inputs, targets = data.batch(indices, device)
        logits, _ = model(inputs)
        losses.append(
            float(
                F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=PAD_TARGET,
                )
            )
        )
        predictions = logits.argmax(-1).cpu().numpy()
        wanted = targets.cpu().numpy()
        for row, conversation in enumerate(indices):
            matches = data.answers[data.answers[:, 0] == conversation]
            for _, start, end, kind, field, _state in matches:
                correct = bool(
                    (predictions[row, start - 1 : end - 1] == wanted[row, start - 1 : end - 1]).all()
                )
                correct_by_conversation.setdefault(conversation, []).append(
                    (int(kind), int(field), correct)
                )

    rows = [row for values in correct_by_conversation.values() for row in values]

    def share(selected: list[tuple[int, int, bool]]) -> float | None:
        return sum(r[2] for r in selected) / len(selected) if selected else None

    result = {
        "val_loss": sum(losses) / len(losses),
        "recall_exact": share([r for r in rows if r[0] == KIND_RECALL]),
        "refusal_exact": share([r for r in rows if r[0] == KIND_REFUSAL]),
    }
    for index, field in enumerate(FIELDS):
        result[f"recall_{field}"] = share(
            [r for r in rows if r[0] == KIND_RECALL and r[1] == index]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs/main"))
    parser.add_argument("--iters", type=int, default=30_000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-conversations", type=int, default=500)
    parser.add_argument("--save-every-seconds", type=float, default=300)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=0, help="CPU threads, 0 for default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    train = Conversations(args.data / "train.npz")
    val = Conversations(args.data / "val.npz")
    config = demo_config()
    model = BDH(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    generator = np.random.default_rng(args.seed)
    step = 0
    best_recall = -1.0

    latest = args.out / "latest.pt"
    if latest.exists():
        payload = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        generator.bit_generator.state = payload["numpy_rng"]
        torch.set_rng_state(payload["torch_rng"].cpu())
        step = payload["step"]
        best_recall = payload["best_recall"]
        print(f"resumed from {latest} at step {step}", flush=True)

    use_autocast = device.type == "cuda"
    autocast_type = (
        torch.bfloat16
        if use_autocast and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=use_autocast and autocast_type == torch.float16
    )
    if use_autocast:
        torch.backends.cuda.matmul.allow_tf32 = True

    parameters = sum(p.numel() for p in model.parameters())
    print(
        f"device={device} parameters={parameters} train={train.count} val={val.count} "
        f"batch={args.batch} iters={args.iters}",
        flush=True,
    )

    def checkpoint() -> None:
        atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": vars(config).copy(),
                "step": step,
                "best_recall": best_recall,
                "numpy_rng": generator.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
            },
            latest,
        )

    def log(record: dict) -> None:
        record = {"step": step, "time": time.time(), **record}
        with open(args.out / "metrics.jsonl", "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    last_save = time.time()
    last_log = time.time()
    last_log_step = step
    running_loss = 0.0
    running_count = 0

    try:
        while step < args.iters:
            model.train()
            rate = learning_rate(step, args.iters, args.lr, args.warmup)
            for group in optimizer.param_groups:
                group["lr"] = rate

            indices = generator.integers(0, train.count, size=args.batch).tolist()
            inputs, targets = train.batch(indices, device)
            with torch.autocast(device.type, dtype=autocast_type, enabled=use_autocast):
                logits, _ = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=PAD_TARGET,
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            running_loss += loss.item()
            running_count += 1

            if step % args.log_every == 0:
                now = time.time()
                log(
                    {
                        "train_loss": running_loss / running_count,
                        "lr": rate,
                        "seconds_per_step": (now - last_log) / (step - last_log_step),
                    }
                )
                last_log, last_log_step = now, step
                running_loss, running_count = 0.0, 0

            if step % args.eval_every == 0 or step == args.iters:
                metrics = evaluate(
                    model, val, args.eval_conversations, args.batch, device
                )
                log(metrics)
                recall = metrics["recall_exact"] or 0.0
                if recall > best_recall:
                    best_recall = recall
                    atomic_save(
                        {
                            "model": model.state_dict(),
                            "config": vars(config).copy(),
                            "step": step,
                            "metrics": metrics,
                        },
                        args.out / "best_model.pt",
                    )
                checkpoint()
                last_save = time.time()
            elif time.time() - last_save >= args.save_every_seconds:
                checkpoint()
                last_save = time.time()
    except KeyboardInterrupt:
        print("interrupted, saving", flush=True)
    finally:
        checkpoint()
        print(f"saved {latest} at step {step}", flush=True)


if __name__ == "__main__":
    main()
