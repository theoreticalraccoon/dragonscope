# Status

Last updated: Mon Nov 2, 2026. Update this before ending any session.

## Frozen: the INFOTEL demo

The demo and its website dashboard are frozen at tag **`infotel-2026-v1`**. Branch **`infotel-2026`** starts from the same commit. Only booth-blocking fixes go on that branch, with no new features, and the tag does not move. The tested package versions are in `requirements-lock.txt`.

To run the frozen version: `git checkout infotel-2026`, then `python -m dragonscope.server`.

## Booth kit

Everything is in `booth/`. Start with `booth/CHECKLIST.md`.

- **Rehearsal: 20 of 20 clean in a real browser**, on the Iris Xe GPU, with off-laptop requests blocked, no page errors and a mean run of 16 s. **Wi-Fi was still on.** The true offline run is one command in the checklist and is still to do.
- **Screens:** no scrolling or clipping at any size tested, from 1280x720 to 4K. At 4K with 100% scaling, chat text is too small, so the checklist says to use 200%. The real booth screen has not been seen.
- **Backup video:** `booth/backup_video.webm`, 47.6 s at 1080p, recorded from the real flow. It is committed on `infotel-2026` only.
- **Printed cards:** `booth/fake_cards.pdf`, with 8 cards. The card numbers are published payment test numbers, as random Luhn-valid numbers could match real cards. Each NIC uses day 999, so it cannot be real. Each number passed 5 of 5 full flows on the trained model, in two conversation shapes.

## Toolkit (on `main`)

DragonScope is an open-source toolkit for BDH developers, a debugger for AI memory, with the bank demo as its flagship example. Toolkit work goes on `main` and leaves the frozen branch alone. The repository is public, under the MIT `LICENSE`, with a description and topics set on GitHub. Version 0.2.0, unreleased; see `CHANGELOG.md`.

Built and tested:

- **API:** `ds.capture(model, ledger=...)`, `MemoryCapture` (`read(text, ledger=None)`, `generate`, `find`, `forget` with `auto`, `subtract` or `rebuild`, `rebuild`, `residual`, `ledger_bytes`, `fork`, `reset`, `neuron_strength`, `history`), `ds.forgetting_report` with `Case` and `Probe`, and `ds.dashboard`. Tested on the demo model, another BDH size, and a separately imported copy of upstream `bdh.py`. A report fails if its probes could not recall the value before forgetting.
- **Ledger modes:** `exact` (default), `compact` (float16, zeros dropped where smaller: at least 2 times smaller, 2.7 on the demo model, about 1e-4 extra error) and `off`. `read(text, ledger=False)` skips entries for text that will never need forgetting. `forget(method="auto")` falls back to `rebuild` for unrecorded bytes.
- **Batched subtract:** one pass over `rho` per 128 erased bytes instead of one per byte. A card number now takes about 5 ms instead of 30, and 320 bytes 34 ms instead of 806.
- **Dashboard:** `dragonscope-dashboard --model models/best_model.pt` or `--untrained`, at http://127.0.0.1:8100. Checked in Edge with reduced motion on, with both models: no page errors, no off-laptop requests, no horizontal scroll at phone width, and after Forget the secret was in neither the page, its inputs, nor its state, including the report form.
- **Benchmarks:** `python -m dragonscope.benchmark`, `python -m dragonscope.benchmark --scale` for the published BDH size, then `python scripts/benchmark_charts.py`. Results and charts are in `docs/benchmark/` and the README. Run them on an idle machine.
- **Long conversations:** `python -m dragonscope.data --max-bytes 2048 --pad-to 2048`, and the `long` run in `notebooks/train_gpu.ipynb`, which now clones the public repo. Default data is unchanged byte for byte. Two training steps ran on CPU. **No long model has been trained**: that is a Colab run.
- **Notebook:** `notebooks/getting_started.ipynb`, executed with outputs saved.
- **Packaging and CI:** `pyproject.toml`, with extras `dashboard`, `charts` and `dev`. A wheel installed outside the repo served every static file and installed the `dragonscope-dashboard` command. Not on PyPI. GitHub Actions runs the full suite on every push to `main`.
- **Docs:** public pages are the README, `docs/EVALUATION.md`, `docs/PROTOCOL.md`, `CONTRIBUTING.md` and `CHANGELOG.md`. Internal notes are in `docs/dev/`.
- **Community post:** a draft only, kept outside git in `notes/announcement_draft.md`. Nothing has been posted.

Demo model, forgetting a 19-byte card, 20 conversations per length:

| Conversation | `subtract` | `rebuild` | Residual | Recall before, after |
|---|---|---|---|---|
| 138 bytes | 4.9 ms | 90 ms | 5.2% | 100%, 0% |
| 428 bytes | 4.9 ms | 318 ms | 4.3% | 60%, 0% |
| 3,208 bytes | 4.0 ms | 5.7 s | 4.0% | not measurable |

Published BDH size (6 layers, 32,768 neurons, 25.3M parameters), random weights: reading takes about 175 ms a byte, `rho` is 201 MB, and the ledger is 793 KB a byte exact or 396 KB compact. At 1,024 bytes, `subtract` took 239 ms and `rebuild` 11.7 s.

Within the trained length, both methods gave the same recall, kept values and replies. Beyond 512 bytes the demo model recalls nothing, so those rows are cost and state only, and replies from the two methods matched in just 22 to 42% of probes. On random weights the residual was about 46%. Do not claim more than this.

## Dashboard

`python -m dragonscope.server`, then open http://127.0.0.1:8000. It is a minimal exhibit: numbered steps and one sentence tell the story, a 3D brain shows memory, and one glowing button shows the next action. Files are in `dragonscope/static/`, with three.js 0.160.0 and its licence bundled locally. The brain replaced the plan's D3 graph, so D3 is gone.

**The model on screen is the real one.** The earlier scope and first brain treated layers as separate neurons and said 16,384. That was wrong: the encoder is shared, so there are **4,096 neurons** (4 heads of 1,024) and each character passes through them in **4 rounds**. About 880 fire per round. Do not reintroduce per-layer neurons.

- **Dots:** one per real neuron. The 4 regions are the 4 heads.
- **Position:** from learned weights (`model_facts` in `dragonscope/activity.py`). Each neuron's firing for every text character gives its tuning, and the two main axes of tuning place it. Measured: similarity to the 5 nearest neighbours is 0.43, against 0.25 to 0.29 for all pairs.
- **Firing:** every neuron's real activation for each round, streamed live as each byte is computed. There is no recording or replay. By default, the server holds each byte for 75 ms when reading and 110 ms when writing so it can be watched, and "Speed: full" removes that. Brightness is the square root of activation, so it stays in order.
- **Lines:** the strongest learned connections, `decoder[i] . encoder[j]`, between neurons firing in consecutive rounds. They are model weights, not `rho`.
- **Red at rest:** the share of each neuron's memory that comes from flagged data, with no threshold. Shares run from 0 to 0.99 with a median of 0.19 when a card is 21% of the session, so it is spread across many neurons rather than stored in one place.
- **Red while active:** a second copy of memory that never read the flagged bytes runs alongside, and each neuron's red is how much its firing differs. This is causal. Over 10 sessions, final-round firing changed by 58% when writing the card's digits, against 4% when answering about the branch, with about 266 neurons changing by over 10% of the peak. The first round changes by exactly 0, as it depends only on the byte. This replaced an earlier indicative measure.
- **Illustrative only:** the brain outline and its folds, and the glow's size and colour. There is no brain stem or cerebellum, by choice. The shell's pulse follows the real firing count and `depends`, and the page says the outline is an illustration.
- **Removed:** the flare on Forget, the red threshold, the boost to weak neurons, and replay.
- **Cannot be made real:** which head sits in which brain region, and the brain shape. The model has neither.

**The red letters are a measurement.** For each written byte, `depends` is how much of its probability disappears in the copy of memory that never read the flagged bytes. Over 15 sessions with a card and NIC stored:

| Question | Digits written, mean `depends` | Share over 0.5 | Other bytes, mean `depends` |
|---|---|---|---|
| Card | 0.88 | 87.5% | 0.05 |
| NIC | 0.89 | 90.9% | 0.10 |
| Branch | n/a | n/a | 0.00 |

Two simpler signals failed and were dropped, so do not bring them back. The share of `rho` from flagged data (0.07 to 0.29) and the attention weight on the card's bytes neither peaked on recalled digits nor separated the card question from the branch question.

Limits: `depends` compares the model with and without **all** flagged data, not one value. A digit can score low when the model would have guessed the same digit anyway. After Forget it is 0 because nothing flagged remains to remove, so it is not separate evidence of erasure.

**The page ignores the reduced-motion preference, on purpose.** This laptop has Windows animations off, which browsers report as `prefers-reduced-motion`. Honouring it skipped all firing and rotation, and test browsers override it, so tests did not show this. The motion is the exhibit, so it always runs. Test with `reduced_motion="reduce"` as well.

Tested in headless Edge at 1366x768, 1920x1080 and phone width: the full flow worked, the secret was not in the page's HTML or JavaScript state after Forget, live frames were dropped when the state arrived, there were no console errors, and the page requested nothing from outside 127.0.0.1. It held 60 frames a second in Edge on the laptop's Iris Xe GPU, and about 35 in Opera GX, the laptop's default browser. Use Edge at the booth, or check GX Control's limits first. **Keep hardware acceleration on in the browser:** with software rendering it fell to about 10 to 16 frames a second and competed with the model for CPU, stalling replies for over a minute. Nobody has yet run it with Wi-Fi off or seen it on a booth screen.

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

`models/best_model.pt` is step 26,500 of a 30,000-step Colab run (about 5 hours on a T4). It has 1.6M parameters. The full run output is in `runs/gpu-main/` on the development laptop, not in git.

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

With live frames and the comparison, a byte took about 45 to 85 ms on battery with the page open. The page shows this live. Before those were added, reading or writing took 17 ms a byte at 4 threads, on battery on Balanced power, and the server uses that setting. Earlier runs measured 5 to 6 ms, probably on mains power, but this has not been compared directly. Replies take under a second either way. The ledger is about 66 KB a byte in memory. **Keep the laptop plugged in at the booth.**

## Next

1. Booth, on `infotel-2026`: run `python scripts/rehearse_browser.py --runs 20` with Wi-Fi off and the laptop plugged in, check the real booth screen, print `booth/fake_cards.pdf`, copy the backup video to USB, and prepare the pitch.
2. Toolkit, after INFOTEL: train the `long` model on Colab and rerun the benchmark with it, to measure recall past 512 bytes. Post the community draft and reach out to BDH users. Publish to PyPI if people want it.

## Known design points

- After Forget, the statement remains with its value gone. Training covered this with `blank` and `masked` conversations, and the evaluation confirms the resulting position gap works.
- `read` needs new bytes after Forget because pending logits are cleared.
- Training output is not in git. Commit milestone models to `models/` only.
- Evaluation runs on the laptop must be detached from the console: a Ctrl-C reached earlier runs. Launch with `DETACHED_PROCESS`, `CREATE_NEW_PROCESS_GROUP` and `CREATE_BREAKAWAY_FROM_JOB`. Results save per case, and rerunning resumes.
