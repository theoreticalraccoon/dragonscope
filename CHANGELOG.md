# Changelog

## 0.2.0 (unreleased)

- Ledger modes: `capture(model, ledger="compact")` stores each byte's entry at least 2 times smaller (2.7 times on the demo model), and `ledger="off"` stores none.
- `read(text, ledger=False)` skips entries for text that will never need forgetting.
- `forget` defaults to `method="auto"`: `subtract` when every byte has an entry, otherwise `rebuild`.
- `python -m dragonscope.benchmark --scale` measures cost and memory at the published BDH size.
- `python -m dragonscope.data --max-bytes 2048 --pad-to 2048` makes longer training conversations.
- Tests run on GitHub Actions.
- `memory.residual()` compares with a batched rebuild, about 10 times faster on long sessions. `residual(stepped=True)` keeps the independent byte-by-byte check.
- `load_model` loads with `weights_only=True`, so a model file cannot run code.
- `forgetting_report` refuses cases with no probe about a forgotten value, which could otherwise pass without testing forgetting.
- The dashboard places neurons in separate regions for any number of heads.

## 0.1.0

- `ds.capture`, `MemoryCapture`, `ds.forgetting_report` and `ds.dashboard`.
- Forget versus rebuild benchmark across conversation lengths.
- Getting-started notebook.
- The bank demo for FITIS INFOTEL 2026, frozen at tag `infotel-2026-v1`.
