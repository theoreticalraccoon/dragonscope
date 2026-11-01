import json
import subprocess
import sys

import numpy as np
import torch

from dragonscope.data import build_dataset
from dragonscope.engine import SessionMemory
from dragonscope.train import load_model


def run_training(data, out, iters):
    subprocess.run(
        [
            sys.executable, "-m", "dragonscope.train",
            "--data", str(data), "--out", str(out), "--iters", str(iters),
            "--batch", "2", "--log-every", "1", "--eval-every", "2",
            "--eval-conversations", "2", "--device", "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_training_saves_resumes_and_loads_into_the_engine(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    np.savez(data / "train.npz", **build_dataset(6, 1, "train"))
    np.savez(data / "val.npz", **build_dataset(2, 2, "val"))
    out = tmp_path / "run"

    run_training(data, out, 2)
    assert torch.load(out / "latest.pt", weights_only=False)["step"] == 2
    assert (out / "best_model.pt").exists()

    run_training(data, out, 3)
    assert torch.load(out / "latest.pt", weights_only=False)["step"] == 3

    records = [json.loads(line) for line in (out / "metrics.jsonl").open()]
    assert [r["step"] for r in records if "train_loss" in r] == [1, 2, 3]
    assert any("recall_exact" in r for r in records)

    model = load_model(out / "latest.pt")
    logits = SessionMemory(model).read(b"U: Hi")
    assert logits.shape == (5, 256)
    assert logits.isfinite().all()
