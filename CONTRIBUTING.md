# Contributing

Issues and pull requests are welcome, especially results from other BDH models and sizes.

## Set up

```bash
pip install -e ".[dashboard,dev]"
python -m pytest -q
```

The tests take about 7 minutes on a laptop CPU.

## Rules

- Do not edit `dragonscope/vendor/bdh/`. It is upstream BDH, kept unmodified with its licence.
- The stateful engine must match the full BDH forward pass, with logits within 1e-3. `tests/test_engine.py` checks this. Run it after any change to `dragonscope/engine.py`.
- Forget must leave no copy of forgotten text anywhere the toolkit keeps data: bytes, ledger, pending logits, history and the dashboard's state.
- Report results as measured. `subtract` leaves a residual, so do not describe it as complete erasure.
- The dashboards must work offline, with no assets loaded from the internet.
- Run benchmarks on an otherwise idle machine, and say what hardware you used.
