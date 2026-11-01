# Status

Last updated: Sun Nov 1, 2026, evening. Update this before ending any session.

The project is ahead of schedule: the Day 1 gates, the card and NIC part of the Day 2 recall gate, the Tuesday Forget go/no-go, the Tuesday backend and the Wednesday dashboard are all done. **Next is Thursday's rehearsal.**

## Dashboard

`python -m dragonscope.server`, then open http://127.0.0.1:8000. It is a minimal exhibit, as the owner asked: one sentence tells the story, memory is a radial scope (rings are layers, quadrants are heads), and one red button forgets. Files are in `dragonscope/static/`, with D3 7.9.0 and its licence bundled locally.

Tested in headless Edge at 1366x768, 1920x1080 and phone width: the full flow worked, the secret was not in the page's HTML or JavaScript state after Forget, there were no console errors, and the page requested nothing from outside 127.0.0.1. Nobody has yet run it with Wi-Fi off, or on a booth screen.

## Backend

`python -m dragonscope.server` serves one session at `ws://127.0.0.1:8000/ws`. `python scripts/booth_check.py --runs 20` runs the booth flow over a real WebSocket on the trained model.

**Measured: 30 of 30 clean runs**, half forgetting a card and half a NIC. In each run the secret was recalled before Forget and refused after it, the other fields checked were kept, and the secret was in no WebSocket frame after Forget. Forget took 0.78 s on average and 0.90 s at worst, on battery. The residual after `subtract` was 3.1 to 5.5%, and a byte-by-byte replay confirmed 0 after the rebuild.

How it handles the limits below:

- **Guided chat only.** There is no free-text message. `say` and `ask` use trained wordings, name and branch come from `split_names("train")`, and an invalid value is refused before the model reads it.
- **Every copy goes.** Forget erases each copy of the value in any format, plus any number in a model reply that is not exactly another value the visitor stated. A reply's flagged span runs from its first digit to its last, once it has 4 or more digits, so garbled echoes are caught. Unit tests use scripted garbled replies, because the booth runs only produced exact echoes.
- **One value per field, per session.** Restating a card after Forget recalled exactly in only 13 of 20 tries, so a new value needs a reset.
- **Session limit of 400 positions.** Forget does not free positions, and replies get less reliable as a session grows. In 20 runs with a 440 limit, one reply at byte 406 got two card digits wrong. In 60 longer sessions, replies ending at bytes 480 to 519 were 93% correct, and those at 600 to 639 were 61% correct. The limit fits four statements, one ask, Forget, a re-ask and at least one other question, and no run in the 30 above gave a wrong reply.
- **Rebuild is now batched.** Stepping byte by byte took about 10 s per Forget on battery, including its check. `SessionMemory.rebuild` now runs each layer over all bytes at once, and `test_rebuild_matches_stepping_byte_by_byte` guards it.

Forget's reach: positions and findings hold no values, so `SessionMemory` is the only server copy, and the deletion log stores only type, last 4 digits and time. Python does not overwrite freed memory, so erased bytes may stay in process RAM until reused. Do not claim otherwise. The dashboard must keep no copy of earlier `state` messages.

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

Reading or writing took 17 ms a byte at 4 threads, on battery on Balanced power, and the server uses that setting. Earlier runs measured 5 to 6 ms, probably on mains power, but this has not been compared directly. Replies take under a second either way. The ledger is about 66 KB a byte in memory. **Keep the laptop plugged in at the booth.**

## Next

1. Rehearse on Thursday: 20 clean offline runs in a row in the browser, with Wi-Fi off and the laptop plugged in. Check readability on the booth screen.
2. Record the backup video, print the fake card numbers, and prepare the pitch and judge questions.

## Known design points

- After Forget, the statement remains with its value gone. Training covered this with `blank` and `masked` conversations, and the evaluation confirms the resulting position gap works.
- `read` needs new bytes after Forget because pending logits are cleared.
- Training output is not in git. Commit milestone models to `models/` only.
- Evaluation runs on the laptop must be detached from the console: a Ctrl-C reached earlier runs. Launch with `DETACHED_PROCESS`, `CREATE_NEW_PROCESS_GROUP` and `CREATE_BREAKAWAY_FROM_JOB`. Results save per case, and rerunning resumes.
