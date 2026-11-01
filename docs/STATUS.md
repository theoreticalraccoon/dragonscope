# Status

Last updated: Sun Nov 1, 2026, afternoon. Update this before ending any session.

The project is ahead of schedule: the Day 1 gates, the card and NIC part of the Day 2 recall gate, and the Tuesday Forget go/no-go are all done. **Next is the backend (Tuesday's build).**

## Model

`models/best_model.pt` is step 26,500 of a 30,000-step Colab run (about 5 hours on a T4). It has 1.6M parameters. The full run output is in `runs/gpu-main/` on the owner's laptop, not in git.

| Field | Exact recall, new values (500-conversation validation) | Exact recall, names seen in training (500-case Forget baseline) |
|---|---|---|
| Card | 100% | 100% |
| NIC | 100% | 100% |
| Name | **0%** | 100% |
| Branch | **0%** | 100% |

Refusal when a value is missing: 100% on validation.

The model truly copies digits, but it memorised the closed list of names and branches and cannot copy new ones. **Decision: no retrain.** The booth flow centres on the card, so the guided chat offers name and branch only from `split_names("train")`.

## Forget evaluation (500 cases, greedy generation)

Results are in `docs/forget_eval_500.json`. Reproduce with `python -m dragonscope.evaluate_forget`, or use shards and `scripts/merge_forget_eval.py`. Half the cases forget a card and half a NIC. Each case uses new random values.

| Check | Pass mark | Before Forget | `subtract` | `replay` |
|---|---|---|---|---|
| Exact recall, trained question | 1% or less | 100% | 0.0% | 0.0% |
| Exact recall, reply forced to start "Your card number is " | 1% or less | 100% | 0.0% | 0.0% |
| Digit match on forced reply (control: unrelated number) | About 10% | 100% | 9.9% (10.1%) | 10.1% (10.8%) |
| Longest shared digit run, worst case (control) | Same as control | 16 | 4 (4) | 4 (4) |
| Other fields kept | Within 5 points | 100% | 100% | 100% |
| "No record" reply, trained question | 90% or more | n/a | 100% | 100% |
| Same, after the model had already repeated the secret, with every copy forgotten | 90% or more | n/a | 100% | 100% |
| Distance from never having seen the secret | Record it | n/a | 3.8% (range 1.8 to 5.4%) | 0 |

**Decision: Forget runs `subtract`, then `rebuild()`.** `subtract` takes about 23 ms. On the 120-byte test session `replay` took about 0.7 s, or roughly 6 ms a byte. The proof panel can show the residual going from about 3.8% to 0. Say plainly that the second step rebuilds memory from the surviving bytes.

### Limits to keep in mind

- **Forget every copy.** In one test, forgetting only the original statement after the model had repeated the number gave `Your card number is 4711 1111 1111 5111.`, which is close to the real `4111 1111 1111 1111`. The backend must scan replies and forget all copies. A garbled echo will not match an exact search, so match on detected spans in replies as well as the literal value.
- **`subtract` does leave a residual,** even though no probe could use it. Do not claim complete erasure from `subtract` alone.
- **Reworded questions are a weak probe.** Before Forget, unseen wordings recalled only 37.6% of secrets, and otherwise the model gave off-topic stock replies. After Forget they recalled 0%, but this says less than the forced-reply test does. The guided chat must use trained question wordings, or visitors will get nonsense.
- **This is not a security proof.** It is a small model tested with these probes on synthetic data.

## Performance on the demo laptop

Reading or writing takes 5 to 6 ms a byte, so replies are instant. The ledger is about 66 KB a byte in memory.

## Next

1. Backend, per the requirements in PLAN.md: FastAPI and WebSocket server around `SessionMemory`, card and NIC detection on input and replies, Forget as `subtract` then `rebuild()` across all copies, and an in-memory deletion log.
2. Define the message format early so the dashboard can be built against it.
3. Dashboard (Wednesday's build).

## Known design points

- After Forget, the statement remains with its value gone. Training covered this with `blank` and `masked` conversations, and the evaluation confirms the resulting position gap works.
- `read` needs new bytes after Forget because pending logits are cleared.
- Training output is not in git. Commit milestone models to `models/` only.
- Evaluation runs on the laptop must be detached from the console: a Ctrl-C reached earlier runs. Launch with `DETACHED_PROCESS`, `CREATE_NEW_PROCESS_GROUP` and `CREATE_BREAKAWAY_FROM_JOB`. Results save per case, and rerunning resumes.
