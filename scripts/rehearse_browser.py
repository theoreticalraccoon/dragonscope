"""Rehearse the booth flow in a real browser, against the real server.

Each run enters a printed fake number, asks for it, forgets it and asks again. A run
fails if recall or refusal is wrong, if the number stays on the page after Forget,
if the neurons do not fire live, or if the page reaches for anything off this laptop.

It requests reduced motion, as Windows does on the demo laptop, and uses installed
Edge, so no browser download is needed. It needs: pip install playwright

Run: python scripts/rehearse_browser.py --runs 20
For the real rehearsal, turn Wi-Fi off first.
"""

import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
REFUSAL = "I don't have that on record."
PAGE_TEXT = (
    "document.documentElement.outerHTML + JSON.stringify(state)"
    " + JSON.stringify(forgetResult) + JSON.stringify(playback)"
)


def wifi_state() -> str:
    """Report what Windows says about Wi-Fi, so the report shows how offline this was."""
    try:
        output = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True, timeout=10
        ).stdout
    except OSError:
        return "unknown"
    for line in output.splitlines():
        parts = line.split(":", 1)
        if parts[0].strip().lower() == "state":
            return parts[1].strip()
    return "no Wi-Fi adapter reported"


def internet_reachable() -> bool:
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=2).close()
        return True
    except OSError:
        return False


def start_server(port: int) -> subprocess.Popen:
    server = subprocess.Popen(
        [sys.executable, "-m", "dragonscope.server", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return server
        except OSError:
            time.sleep(0.5)
    server.kill()
    raise RuntimeError("the server did not start within 2 minutes")


def until(page, expression: str, seconds: float = 60) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if page.evaluate(expression):
            return
        page.wait_for_timeout(50)
    raise TimeoutError(expression)


def exchange(page, action) -> int:
    """Run one exchange, returning the most neurons seen firing while it was live."""
    action()
    until(page, "playback !== null")
    most = 0
    while page.evaluate("playback !== null"):
        shown = page.evaluate("document.querySelector('.hud-stat.live b')?.textContent ?? ''")
        if shown:
            most = max(most, int(shown.replace(",", "")))
        page.wait_for_timeout(100)
    until(page, "busy === false")
    return most


def last_line(page) -> str:
    return page.inner_text("#transcript").strip().split("\n")[-1].strip()


def one_run(page, entry: dict) -> list[str]:
    problems = []
    field, value = entry["field"], entry["value"]
    answer = {"card": "Your card number is {}.", "nic": "Your NIC is {}."}[field].format(value)

    page.click("#reset")
    until(page, "busy === false && state.turns.length === 0")
    if field == "nic":
        page.click("#tabs button:has-text('NIC')")
    page.fill("#say-value", value)
    if not exchange(page, lambda: page.press("#say-value", "Enter")):
        problems.append("no firing while storing")
    if "remembers your" not in page.inner_text("#headline"):
        problems.append("value not flagged")

    if not exchange(page, lambda: page.click("#ask-now")):
        problems.append("no firing while recalling")
    if last_line(page) != answer:
        problems.append(f"wrong recall: {last_line(page)!r}")

    page.click(".act.forget")
    until(page, "busy === false && storyActive")
    text = page.evaluate(PAGE_TEXT)
    if value in text or value.replace(" ", "") in text:
        problems.append("the number stayed on the page after Forget")
    result = page.evaluate("forgetResult")
    if result["residual_after_rebuild"] > 1e-5:
        problems.append("memory not rebuilt")
    if result["checks"]["ledger_entries_left"] or result["checks"]["value_in_memory_bytes"]:
        problems.append(f"traces left: {result['checks']}")

    if not exchange(page, lambda: page.click("#ask-again")):
        problems.append("no firing while asking again")
    if last_line(page) != REFUSAL:
        problems.append(f"no refusal: {last_line(page)!r}")
    if page.inner_text("#headline") != "It no longer knows.":
        problems.append(f"wrong headline: {page.inner_text('#headline')!r}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--block-network", action="store_true", help="also block off-laptop requests in the browser")
    parser.add_argument("--report", type=Path, default=ROOT / "booth" / "rehearsal_report.md")
    args = parser.parse_args()

    numbers = json.loads((ROOT / "booth" / "fake_numbers.json").read_text(encoding="utf-8"))
    entries = [{"field": k, "value": v} for pair in numbers for k, v in pair.items()]
    wifi, online = wifi_state(), internet_reachable()
    print(f"Wi-Fi: {wifi}. Internet reachable: {online}.", flush=True)

    server = start_server(args.port)
    errors, outside, results = [], [], []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                channel="msedge",
                headless=False,
                args=["--no-first-run", "--ignore-gpu-blocklist"],
            )
            context = browser.new_context(
                viewport={"width": 1366, "height": 768}, reduced_motion="reduce"
            )
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

            def watch(route):
                url = route.request.url
                if "127.0.0.1" in url or url.startswith("data:"):
                    route.continue_()
                    return
                outside.append(url)
                if args.block_network:
                    route.abort()
                else:
                    route.continue_()

            context.route("**/*", watch)
            page.goto(f"http://127.0.0.1:{args.port}/")
            page.wait_for_selector("#say-value")
            renderer = page.evaluate(
                "(() => { const g = document.createElement('canvas').getContext('webgl2');"
                " const e = g && g.getExtension('WEBGL_debug_renderer_info');"
                " return e ? g.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'unknown'; })()"
            )
            for index in range(args.runs):
                entry = entries[index % len(entries)]
                started = time.time()
                try:
                    problems = one_run(page, entry)
                except Exception as error:  # A stuck page counts as a failed run.
                    problems = [f"{type(error).__name__}: {str(error)[:160]}"]
                    page.reload()
                    page.wait_for_selector("#say-value")
                seconds = time.time() - started
                results.append((entry, problems, seconds))
                status = "ok" if not problems else "; ".join(problems)
                print(f"run {index + 1:>2} {entry['field']:4} {seconds:5.1f}s: {status}", flush=True)
            browser.close()
    finally:
        server.terminate()

    clean = sum(not problems for _, problems, _ in results)
    # The rehearsal asks for clean runs in a row, so find the longest streak.
    streak = best = 0
    for _, problems, _ in results:
        streak = 0 if problems else streak + 1
        best = max(best, streak)
    lines = [
        "# Rehearsal report",
        "",
        f"Time: {datetime.now().isoformat(timespec='seconds')}",
        f"Clean runs: **{clean} of {len(results)}**. Longest clean streak: **{best}**.",
        f"Wi-Fi: {wifi}. Internet reachable: {online}. Off-laptop requests blocked in the browser: {args.block_network}.",
        f"Graphics: {renderer}",
        f"Page errors: {len(errors)}. Requests leaving the laptop: {len(outside)}.",
        f"Mean run: {sum(r[2] for r in results) / len(results):.1f} s. Slowest: {max(r[2] for r in results):.1f} s.",
        "",
    ]
    lines += [f"- Error: {error[:200]}" for error in errors[:10]]
    lines += [f"- Left the laptop: {url[:200]}" for url in outside[:10]]
    lines += [
        f"- Run {index + 1} ({entry['field']}): {'; '.join(problems)}"
        for index, (entry, problems, _) in enumerate(results)
        if problems
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    passed = clean == len(results) and not errors and not outside
    print("PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
