"""Run the gateway's review console.

    python -m dragonscope.gateway                      # offline, with the trained model
    python -m dragonscope.gateway --connector anthropic  # sends cleared text to Claude
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import uvicorn

from dragonscope.gateway.audit import AuditLog
from dragonscope.gateway.connector import make_connector
from dragonscope.gateway.gate import LeakGate
from dragonscope.gateway.pipeline import Gateway
from dragonscope.gateway.server import create_app
from dragonscope.model_config import demo_config
from dragonscope.train import load_model
from dragonscope.vendor.bdh.bdh import BDH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", type=Path, help="a saved model, from dragonscope.train")
    source.add_argument("--untrained", action="store_true", help="random weights, for trying it")
    parser.add_argument("--connector", choices=("stub", "anthropic"), default="stub")
    parser.add_argument(
        "--block-on-memory", action="store_true",
        help="let the experimental memory check block requests, not only report",
    )
    parser.add_argument("--audit-file", type=Path, help="also write the audit log here, as JSONL")
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()

    if args.untrained:
        torch.manual_seed(0)
        model: BDH = BDH(demo_config()).eval()
    else:
        model = load_model(args.model or Path("models/best_model.pt"))
    gateway = Gateway(
        model,
        connector=make_connector(args.connector),
        gate=LeakGate(model, block_on_memory=args.block_on_memory),
        audit=AuditLog(args.audit_file),
    )
    print(f"DragonScope gateway ({args.connector}): http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(
        create_app(gateway),
        host="127.0.0.1",
        port=args.port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
