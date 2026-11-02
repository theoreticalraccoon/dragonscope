"""Record the real booth flow, for use if the live demo fails on the day.

It needs: pip install playwright, then python -m playwright install ffmpeg
Run: python scripts/record_backup_video.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from rehearse_browser import start_server, until  # noqa: E402

PORT = 8791
CARD = "4242 4242 4242 4242"
SIZE = {"width": 1920, "height": 1080}


def main() -> None:
    server = start_server(PORT)
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder, sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                channel="msedge", headless=False, args=["--ignore-gpu-blocklist"]
            )
            context = browser.new_context(
                viewport=SIZE, record_video_dir=folder, record_video_size=SIZE, reduced_motion="reduce"
            )
            page = context.new_page()
            page.goto(f"http://127.0.0.1:{PORT}/")
            page.wait_for_selector("#say-value")
            page.wait_for_timeout(3500)
            page.locator("#say-value").press_sequentially(CARD, delay=70)
            page.wait_for_timeout(700)
            page.press("#say-value", "Enter")
            until(page, "playback !== null")
            until(page, "busy === false")
            page.wait_for_timeout(3500)
            page.click("#ask-now")
            until(page, "playback !== null")
            until(page, "busy === false")
            page.wait_for_timeout(4500)
            page.click(".act.forget")
            until(page, "busy === false && storyActive")
            page.wait_for_timeout(4500)
            page.click("#ask-again")
            until(page, "playback !== null")
            until(page, "busy === false")
            page.wait_for_timeout(6000)
            video = page.video
            context.close()
            target = ROOT / "booth" / "backup_video.webm"
            video.save_as(target)
            browser.close()
            print(f"{target}: {target.stat().st_size / 1e6:.1f} MB")
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
