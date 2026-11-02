# Draft: post for pathwaycom/bdh GitHub Discussions

Not posted. Before posting, rerun `python -m dragonscope.benchmark` if the code has changed.

Suggested category: Show and tell.

---

**Title:** DragonScope: inspect and erase what a BDH model's memory holds

Hi all. I built a small open-source toolkit for looking at, and editing, the memory BDH builds up while it reads. I'd like feedback from people working with BDH, especially on whether this is useful beyond my example.

**What it does**

- `ds.capture(model)` wraps a `BDH` model from this repo and records each byte's effect on the state `rho`, with neuron firing per round.
- `memory.forget("some text")` removes every copy of that text from memory, either by subtracting each byte's recorded contribution (`subtract`) or by rereading the session without it (`rebuild`).
- `ds.forgetting_report(model, cases)` checks your own probes before forgetting, after `subtract` and after `rebuild`. It fails if the probes could not recall the value beforehand.
- A local dashboard shows the real neurons firing, and which ones change because of text you choose to watch, measured against a copy of memory that never read it.

Example: a small bank assistant (1.6M parameters, trained on synthetic chat) that remembers a card number, then forgets it.

**What I measured**

Forgetting a 19-byte card number, 20 conversations per length, laptop CPU:

| Conversation | `subtract` | `rebuild` | Residual after `subtract` |
|---|---|---|---|
| 138 bytes | 30 ms | 65 ms | 5.2% |
| 428 bytes | 33 ms | 287 ms | 4.3% |
| 3,208 bytes | 60 ms | 6.5 s | 4.0% |

Where the model could recall the card, both methods took recall from 100% to 0%, kept other values, and gave the same replies. `subtract` leaves a residual of about 4 to 6% of the state from the second layer on, and `rebuild` leaves none. `subtract` also needs a ledger of each byte's contribution, about 50 to 70 KB a byte for this model.

Limits: the model only recalls within its 512-byte training length, so longer rows measure cost and state, not recall. On random weights the residual was about 46%, so it depends on the model. This is not machine unlearning (weights never change) and not a security proof.

**Questions for you**

1. Is a byte-level memory ledger practical at your model sizes, or would you want a different way to subtract?
2. Are there memory behaviours you'd want to see in the dashboard that it doesn't show?
3. Has anyone measured how much of `rho` a single token contributes in larger BDH models?

Code, benchmark and a getting-started notebook: https://github.com/theoreticalraccoon/dragonscope

Thanks to the BDH authors for releasing the code. DragonScope vendors it unmodified, with its MIT licence.
