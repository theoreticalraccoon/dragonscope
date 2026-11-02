# DragonScope

A debugger for the memory of BDH (Dragon Hatchling) models. It records what a model's memory holds while it reads, shows it live, erases chosen text from it, and measures whether the erasure worked.

BDH keeps a session's context in one fixed-size state, `rho`, which the paper treats as a compressed form of the synaptic state. DragonScope makes that state something you can inspect and edit:

- **Capture:** wrap a model and record each byte's effect on memory, with neuron firing per round.
- **Memory map:** a live dashboard of the real neurons firing, the text in memory, and which neurons change because of text you choose to watch.
- **Forget:** remove every copy of a value from memory, by targeted subtraction or by rereading the session without it.
- **Forgetting report:** test your own probes before forgetting, after `subtract` and after `rebuild`, and record the residual and cost.

The flagship example is a bank assistant that remembers a card number and then forgets it, built for FITIS INFOTEL 2026.

All conversation data is synthetic. Never type real card or ID numbers into it.

## Quick start

```bash
pip install -e ".[dashboard]"
```

```python
import dragonscope as ds

model = ds.load_model("models/best_model.pt")   # or any BDH model from pathwaycom/bdh
memory = ds.capture(model)
memory.read("U: My card number is 4242 4242 4242 4242.\nA: Noted.\n")

edit = memory.forget("4242 4242 4242 4242")      # method="subtract" by default
print(edit.copies, edit.bytes_erased, edit.seconds, edit.residual)

memory.forget("Noted.", method="rebuild")        # exact, but rereads the session
ds.dashboard(memory)                             # http://127.0.0.1:8100
```

Or open the dashboard from a terminal: `dragonscope-dashboard --model models/best_model.pt`, or `--untrained` to try it with random weights.

[notebooks/getting_started.ipynb](notebooks/getting_started.ipynb) walks through capture, forgetting and a report in about a minute on a laptop CPU, with its outputs saved.

## API

| Call | What it does |
|---|---|
| `ds.capture(model, keep_states=False)` | Wraps a BDH model. Returns a `MemoryCapture`. `keep_states` copies `rho` at every byte, about 8 MB a byte for the demo model. |
| `memory.read(text)`, `memory.generate(max_bytes)` | Reads text, or writes greedily, recording a `Step` per byte in `memory.history`. |
| `memory.find(text)` | The positions of every copy of the text in memory. |
| `memory.forget(target, method)` | `target` is text, bytes or a `(start, end)` range. Returns a `ForgetResult` with copies, bytes erased, time and residual per round. Forgotten bytes leave the history too. |
| `memory.rebuild()`, `memory.residual()` | Rereads the surviving text into a fresh memory, and measures the distance from doing so. |
| `memory.fork()`, `memory.reset()` | An independent copy, and an empty session. |
| `memory.neuron_strength()` | How much memory each neuron holds, per head. |
| `ds.forgetting_report(model, cases)` | Runs `ds.Case(text, forget, probes)` cases with `ds.Probe(prompt, expect, about)`. Returns a `ForgettingReport` with `passed`, `to_markdown()`, `to_dict()` and `save()`. |
| `ds.dashboard(memory, port=8100)` | Opens the dashboard for a session in the background. |

A report fails if its probes could not recall the value before forgetting, so a model that never learned the value cannot earn a pass.

## Forget versus rebuild

Rereading the session without the value is perfect forgetting by definition. So the targeted edit has to answer two questions: how close does it get, and how much cheaper is it? Results are from `python -m dragonscope.benchmark`, 20 conversations per length, on the demo model on an i5-1335U laptop at 4 threads. Raw numbers are in [docs/benchmark/results.json](docs/benchmark/results.json).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmark/time_by_length_dark.png">
  <img alt="Time to forget a 19-byte card number. Subtract stays between 30 and 60 ms from 138 to 3,208 bytes of conversation. Rebuild grows from 65 ms to 6.5 s." src="docs/benchmark/time_by_length_light.png" width="720">
</picture>

| Conversation | `subtract` | `rebuild` | Faster by | Residual after `subtract` | Recall before, after | Kept | Ledger |
|---|---|---|---|---|---|---|---|
| 138 bytes | 30 ms | 65 ms | 2.2x | 5.2% | 100%, 0% | 100% | 7 MB |
| 221 bytes | 36 ms | 135 ms | 3.7x | 4.4% | 100%, 0% | 100% | 16 MB |
| 428 bytes | 33 ms | 287 ms | 8.7x | 4.3% | 60%, 0% | 85% | 27 MB |
| 855 bytes | 33 ms | 708 ms | 21x | 5.7% | not measurable | | 56 MB |
| 1,646 bytes | 35 ms | 1.9 s | 55x | 5.3% | not measurable | | 110 MB |
| 3,208 bytes | 60 ms | 6.5 s | 108x | 4.0% | not measurable | | 221 MB |

What this shows:

- **Cost.** `subtract` is roughly flat with conversation length, and `rebuild` grows faster than linearly. The gap is small for short sessions and large for long ones.
- **Accuracy.** Wherever the model could recall the card, `subtract` removed it as fully as `rebuild` did, kept the other value as well, and gave identical replies. Its residual is 4 to 6% of the state from the second round on, and 0 in the first. `rebuild` leaves a residual under 1e-6.
- **Limits.** The model was trained on conversations of up to 512 bytes and cannot recall anything beyond that, so longer rows measure cost and state only. There, `subtract` and `rebuild` gave the same reply in only 22 to 42% of probes, because the residual shifts an already unreliable model. The residual is not small for every model: on random weights it was about 46%.
- **Memory.** `subtract` needs a ledger of each byte's contribution, 50 to 70 KB a byte here, which grows with the session while `rho` stays at 8.4 MB.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmark/time_by_erased_size_dark.png">
  <img alt="In a 1,646-byte conversation, subtract takes 11 ms for 5 bytes and 806 ms for 320 bytes. Rebuild takes about 1.9 s whatever the size." src="docs/benchmark/time_by_erased_size_light.png" width="720">
</picture>

`subtract` costs about 2.5 ms per erased byte, so it scales with what is erased rather than with the whole history, and stays cheaper than rebuilding until a large share of the session is erased.

**Use `subtract` for speed, then `rebuild` when you need no residual.** The bank demo does both. None of this is a security proof or machine unlearning: the weights never change, and the evidence covers these probes on this model.

## The bank demo

```bash
python -m dragonscope.server
```

Then open http://127.0.0.1:8000. A visitor gives a fake card number, asks for it back, forgets it, and asks again. The version frozen for FITIS INFOTEL 2026 is tag `infotel-2026-v1`, on branch `infotel-2026`, with tested package versions in `requirements-lock.txt`. The booth kit is in `booth/`.

On a 500-case evaluation, exact recall of the forgotten card or NIC went from 100% to 0% with both methods, and other fields stayed at 100%. Details are in [docs/STATUS.md](docs/STATUS.md).

## Train the demo model

```bash
pip install -r requirements.txt
python -m dragonscope.data --out data
python -m dragonscope.train --data data --out runs/main
```

Training resumes from `latest.pt` if you rerun the same command. On a CPU it is slow, so use [notebooks/train_gpu.ipynb](notebooks/train_gpu.ipynb) on Colab or Kaggle.

## Tests

```bash
python -m pytest -q
```

The stateful engine is checked against the unmodified BDH forward pass, with logits within 1e-3.

## Layout

| Path | Purpose |
|---|---|
| `dragonscope/toolkit.py` | `capture` and `MemoryCapture` |
| `dragonscope/report.py` | Forgetting reports |
| `dragonscope/dashboard.py` | The toolkit dashboard, for any session |
| `dragonscope/benchmark.py` | Forget versus rebuild across conversation lengths |
| `dragonscope/engine.py` | Stateful inference, the memory `rho`, and Forget |
| `dragonscope/activity.py` | Records which neurons really fire for each byte |
| `dragonscope/server.py`, `session.py`, `detect.py` | The bank demo |
| `dragonscope/data.py`, `train.py`, `evaluate_forget.py` | Synthetic data, training and evaluation |
| `dragonscope/static/` | Both pages and the 3D brain, with three.js bundled locally |
| `dragonscope/vendor/bdh/` | Unmodified upstream BDH, with its licence |
| `scripts/` | Booth checks, rehearsal and benchmark charts |
| `docs/` | Status, plan, protocol and benchmark results |

## Licence

DragonScope is under the MIT licence in [LICENSE](LICENSE).

`dragonscope/static/vendor/three.min.js` is three.js 0.160.0, under the licence in [three-LICENSE.txt](dragonscope/static/vendor/three-LICENSE.txt).

`dragonscope/vendor/bdh/` is from [pathwaycom/bdh](https://github.com/pathwaycom/bdh) at commit `2b0d7a45b058d4309c84a10e0768d541fe18bdc2`. It is copyright 2025 Pathway Technology, Inc., under the MIT licence in [its LICENSE.md](dragonscope/vendor/bdh/LICENSE.md).

The paper is Kosowski et al., "The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain", [arXiv:2509.26507](https://arxiv.org/abs/2509.26507).
