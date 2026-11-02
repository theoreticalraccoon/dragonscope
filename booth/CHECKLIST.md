# Booth checklist

For the person running the DragonScope demo at FITIS INFOTEL 2026.

## Before leaving

- [ ] Print `fake_cards.pdf` on A4 at 100% scale and cut out the cards. Print several copies.
- [ ] Copy `backup_video.webm` to a USB drive, and check that it plays on the booth laptop.
- [ ] Run the true offline rehearsal below, and keep the report.
- [ ] Pack the laptop's charger.

## True offline rehearsal

The automated run so far blocked outside requests but left Wi-Fi on. This is the real test:

1. Turn Wi-Fi off, and unplug any network cable.
2. Plug in the charger.
3. Run:

   ```bash
   python scripts/rehearse_browser.py --runs 20
   ```

4. It should print `PASS`, with `Internet reachable: False` in `booth/rehearsal_report.md`. It takes about 6 minutes, and a browser window will open.

## At the booth

1. Keep the laptop plugged in, and use **Edge**. Keep hardware acceleration on in its settings.
2. Run the frozen version:

   ```bash
   git checkout infotel-2026
   python -m dragonscope.server
   ```

3. Open http://127.0.0.1:8000 and press F11 for full screen.
4. Check the screen from where visitors will stand:
   - **1366x768 to 1920x1080:** should need no changes.
   - **2560x1440:** the left column's text may look small. Press Ctrl and + until it reads comfortably.
   - **4K:** set Windows display scaling to 200%. At 100%, chat text is too small.
5. Run one full flow yourself, then click **Reset**.

## Each visitor

1. Give them a printed card, and have them type the card number.
2. Point out the brain storing it and the red neurons.
3. Click the glowing question, and point out the red digits and the meter.
4. Click **Forget**.
5. Click **Ask it again**. It should reply "I don't have that on record."
6. Click **Start over**.

Use the printed numbers, not "make one up". Random numbers could by chance match a real card. Never type a real card or NIC.

## If something goes wrong

| Problem | What to do |
|---|---|
| Replies stall or the brain looks choppy | Check that hardware acceleration is on, and close other programs |
| "This session is full" | Click **Reset** |
| A wrong recall after long use | Click **Reset**. Reliability falls as a session grows |
| Page looks out of date or broken | Press Ctrl+F5 |
| Server window closed or crashed | Run `python -m dragonscope.server` again |
| Live demo won't recover | Play `backup_video.webm` and say it is a recording |

## What to say

Say it erases session memory in a prototype, and that the brain outline is an illustration of real neuron activity, slowed to watch. Do not claim regulatory compliance, production scale, machine unlearning or a security proof.
