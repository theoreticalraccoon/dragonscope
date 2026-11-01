# DragonScope plan

Deadline: demo-ready by the night of Thu Nov 5, 2026, for FITIS INFOTEL 2026 on Nov 6.
Pass marks are working targets. Live progress is in [STATUS.md](STATUS.md).

## Booth demo

A visitor chats as a bank customer and shares a printed fake card number. The dashboard flags it and shows where it is stored. They click Forget, ask for the number again, and the AI cannot recall it.

## Schedule

| Day | Build | Must be true by end of day |
|---|---|---|
| Sun Nov 1: engine and data | Repo and BDH with MIT notice. Stateful engine and Forget v1 tested on an untrained model. 50k+ conversations. Start training. | Stateful and full-forward logits match (max difference under 1e-3). Training is running and saving checkpoints. |
| Mon Nov 2: recall and Forget | Train, then run Forget on the trained model. Evaluate with the ledger. | At least 90% exact recall on 500 unseen conversations by 6 pm. First Forget results by night. |
| Tue Nov 3: verdict and backend | Finish Forget tests by 1 pm and choose a method. FastAPI and WebSocket server, card and NIC detection, fixed message format. | Go/no-go at 1 pm. Backend works from a test script. |
| Wed Nov 4: dashboard | Guided chat, D3 graph, red alert, Forget button, "ask it again", proof panel. | Full booth flow works. No new features after tonight. |
| Thu Nov 5: rehearse | Wi-Fi off, bug fixes, backup video, pitch and judge questions. Pack by 9 pm. | 20 clean offline runs in a row. |

## Forget evaluation (Tue 1 pm decision)

Run on 500 conversations, comparing A (`subtract`) with B (`replay`).

| Check | Pass mark |
|---|---|
| Exact recall of the forgotten field | 1% or less |
| Digit match on the forgotten field | About 10% (chance) |
| Recall of other fields | Within 5 points of pre-Forget |
| Correct "no record" reply | 90% or more |

Also try reworded extraction questions, and record the distance between A and B with `residual_from_replay`.

Fallbacks, in order:

1. A passes: present surgical deletion.
2. A fails: use B, and say plainly that it rebuilds memory without the secret.
3. Multi-field recall fails: demo the card number only.
4. Forget fails entirely: show the memory map and present Forget as future work.

## Backend and dashboard requirements

- Detect cards with Luhn validation, and NICs as 9 digits plus V/X or 12 digits. Scan model replies too.
- Forget must also clear the ledger, chat display, logs and WebSocket state.
- Keep everything in memory and bind to `127.0.0.1`.
- Draw only the top 150 or so neurons, on a canvas.
- The proof panel shows the memory change, the re-ask result and a deletion log entry.
- The deletion log stores only data type, last 4 digits and time. Never store a hash of a card number, because it is easy to brute-force.
- Bundle D3 locally, use greedy output, add a reset button, and give visitors printed fake numbers.

## Claims to avoid

Say it erases session memory in a prototype. The weights only saw synthetic data, so this is not machine unlearning. Do not claim regulatory compliance or production scale.

## Open items for the owner

- Confirm INFOTEL slot, power, screen and submission requirements with FITIS.
- Settle the name spelling before printing: "DragonScope" is used here, and an earlier message said "Dragonacope".
