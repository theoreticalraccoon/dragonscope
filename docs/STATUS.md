# Status

Last updated: Sun Nov 1, 2026. Update this before ending any session.

## Done (Day 1)

- BDH vendored unmodified with its MIT licence.
- `SessionMemory` in `dragonscope/engine.py`: stateful reads, greedy generation, ledger, Forget by `subtract` or `replay`.
- Stateful and full-forward logits match on untrained models: max difference 6.0e-07 on the demo config and 1.1e-06 on the upstream default config. The gate is under 1e-3.
- 60,000 training and 2,000 validation conversations, regenerated deterministically by `python -m dragonscope.data --out data`.
- Resumable training with exact-recall evaluation.

## Measured so far (untrained model, one sample conversation)

| Forget method | Relative distance from a session that never saw the secret |
|---|---|
| `replay` | 0 |
| `subtract` | 0.42 overall: layer 0 is 2.6e-07, layers 1 to 3 are 0.41 to 0.47 |

This is the residual predicted in the plan. Bytes after the secret absorbed it at deeper layers. Whether it lets the trained model recall the secret is Monday's question, and an untrained model cannot answer it.

## Training

- CPU run on the owner's laptop: `%USERPROFILE%\dragonscope-runs\cpu-main`, about 9 s/step at batch 16, so roughly 400 steps an hour. It is too slow to reach the Monday gate by itself.
- GPU run: not started yet. `notebooks/train_gpu.ipynb` is written for Colab but has not been run there.
- Training output is not in git, so cloud sessions cannot see it until a milestone model is committed to `models/`.

## Next (Day 2)

1. Get a GPU run going and watch `recall_exact` and `refusal_exact`.
2. Write `dragonscope/evaluate_forget.py` for the checks in PLAN.md. It can be written and tested on an untrained model while training runs.
3. Commit a milestone `best_model.pt` to `models/` as soon as recall is useful.

If recall stalls below the gate, try these in order: train on one field first, shorten conversations, give answer bytes more weight in the loss, then enlarge the model.

## Known design points

- After Forget, the statement remains with its value gone. Training covers this with `blank` and `masked` conversations, but `subtract` and `replay` leave a position gap that training never saw, so test that directly.
- `read` always requires new bytes after Forget because pending logits are cleared.
- The ledger costs about 66 KB per byte on the demo config.
