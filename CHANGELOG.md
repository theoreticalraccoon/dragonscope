# Changelog

## 0.2.0 (unreleased)

- Ledger modes: `capture(model, ledger="compact")` stores each byte's entry at least 2 times smaller (2.7 times on the demo model), and `ledger="off"` stores none.
- `read(text, ledger=False)` skips entries for text that will never need forgetting.
- `forget` defaults to `method="auto"`: `subtract` when every byte has an entry, otherwise `rebuild`.
- `python -m dragonscope.benchmark --scale` measures cost and memory at the published BDH size.
- `python -m dragonscope.data --max-bytes 2048 --pad-to 2048` makes longer training conversations.
- Tests run on GitHub Actions.

## 0.1.0

- `ds.capture`, `MemoryCapture`, `ds.forgetting_report` and `ds.dashboard`.
- Forget versus rebuild benchmark across conversation lengths.
- Getting-started notebook.
- The bank demo for FITIS INFOTEL 2026, frozen at tag `infotel-2026-v1`.
