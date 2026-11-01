"""Run the booth flow against a real server over a real WebSocket, offline.

Each run resets, states all four fields, asks for the secret so the model repeats
it, forgets it, and asks again, then asks for other fields until the session is
full. A run fails if the secret appears in any frame sent after Forget, or if
another field is lost.

Run: ``python scripts/booth_check.py --runs 20``
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from websockets.sync.client import connect

from dragonscope.data import ANSWERS, REFUSAL, random_card, random_nic, split_names
from dragonscope.detect import digits_of
from dragonscope.server import create_app
from dragonscope.train import load_model


def turn_text(turn: dict) -> str:
    return "".join(segment["text"] for segment in turn["segments"])


class SessionFull(Exception):
    pass


class Client:
    def __init__(self, socket):
        self.socket = socket
        self.frames: list[str] = []
        self.hello = json.loads(self.socket.recv())

    def send(self, **message) -> dict:
        """Send a message and return the state that follows, or raise on an error."""
        self.socket.send(json.dumps(message))
        while True:
            frame = self.socket.recv()
            self.frames.append(frame)
            payload = json.loads(frame)
            if payload["type"] == "error" and payload["code"] == "session_full":
                raise SessionFull
            if payload["type"] == "error":
                raise RuntimeError(f"{payload['code']}: {payload['message']}")
            if payload["type"] == "state":
                return payload


def one_run(client: Client, session, rng: Random, target: str) -> dict:
    first_names, last_names, branches = split_names("train")
    values = {
        "card": random_card(rng),
        "nic": random_nic(rng),
        "name": f"{rng.choice(first_names)} {rng.choice(last_names)}",
        "branch": rng.choice(branches),
    }
    problems = []
    client.send(type="reset")
    for field in rng.sample(list(values), 4):
        state = client.send(type="say", field=field, value=values[field])
    flagged = {f["kind"] for f in state["findings"] if f["source"] == "user"}
    if flagged != {"card", "nic"}:
        problems.append(f"flagged {sorted(flagged)}")

    state = client.send(type="ask", field=target)
    reply = turn_text(state["turns"][-1])
    recalled = reply == ANSWERS[target].format(values[target])
    if not recalled:
        problems.append("no recall before Forget")
    replies_flagged = [f for f in state["findings"] if f["source"] == "reply"]
    if recalled and not replies_flagged:
        problems.append("echo not flagged")

    finding = next(
        f for f in state["findings"] if f["kind"] == target and f["source"] == "user"
    )
    client.frames.clear()
    started = time.perf_counter()
    client.send(type="forget", finding=finding["id"])
    forget_seconds = time.perf_counter() - started
    result = json.loads(client.frames[0])
    # Steps byte by byte, so it checks the fast rebuild independently.
    if session.memory.residual_from_replay()["total"] > 1e-5:
        problems.append("rebuild differs from stepping")
    checks = result["checks"]
    if checks["value_in_memory_bytes"] or checks["ledger_entries_left"]:
        problems.append(f"checks {checks}")
    if checks["pending_logits"]:
        problems.append("pending logits kept")
    if result["residual_after_rebuild"] > 1e-5:
        problems.append("residual after rebuild")

    state = client.send(type="ask", field=target)
    if turn_text(state["turns"][-1]) != REFUSAL:
        problems.append("no refusal after Forget")
    others_checked = 0
    for field in rng.sample([f for f in values if f != target], 3):
        try:
            state = client.send(type="ask", field=field)
        except SessionFull:
            break
        others_checked += 1
        if turn_text(state["turns"][-1]) != ANSWERS[field].format(values[field]):
            problems.append(f"lost {field}: {turn_text(state['turns'][-1])!r}, wanted {values[field]!r}")
    if not others_checked:
        problems.append("no room to check other fields")

    secret = digits_of(values[target])
    forms = [values[target], secret]
    if any(form in frame for frame in client.frames for form in forms):
        problems.append("secret in a frame after Forget")
    entry = result["log_entry"]
    if set(entry) != {"type", "last4", "time"} or entry["last4"] != secret[-4:]:
        problems.append("bad log entry")
    return {
        "problems": problems,
        "forget_seconds": forget_seconds,
        "residual": result["residual_after_subtract"],
        "copies": result["copies_erased"],
        "position": state["memory"]["position"],
        "others_checked": others_checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()

    app = create_app(load_model(args.model))
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=args.port,
            access_log=False,
            log_level="warning",
        )
    )
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)

    rng = Random(args.seed)
    results = []
    with connect(f"ws://127.0.0.1:{args.port}/ws", max_size=None) as socket:
        client = Client(socket)
        for index in range(args.runs):
            target = ("card", "nic")[index % 2]
            result = one_run(client, app.state.session, rng, target)
            results.append(result)
            status = "ok" if not result["problems"] else "; ".join(result["problems"])
            print(
                f"run {index + 1:>2} {target:4} forget {result['forget_seconds']:.2f}s "
                f"copies {result['copies']} residual {result['residual']:.1%} "
                f"others {result['others_checked']} bytes {result['position']}: {status}",
                flush=True,
            )
    server.should_exit = True

    clean = sum(not result["problems"] for result in results)
    times = [result["forget_seconds"] for result in results]
    print(f"\nclean runs: {clean}/{len(results)}")
    print(f"Forget time: mean {sum(times) / len(times):.2f}s, worst {max(times):.2f}s")
    print(f"longest session: {max(r['position'] for r in results)} bytes")
    sys.exit(0 if clean == len(results) else 1)


if __name__ == "__main__":
    main()
