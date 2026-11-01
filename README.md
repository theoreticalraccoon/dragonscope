# DragonScope

An AI assistant whose memory you can see and erase, built on the Dragon Hatchling (BDH) architecture. A dashboard will show what the model remembered from a conversation, flag sensitive data such as card and NIC numbers, and delete a chosen memory in one click. It is a prototype for FITIS INFOTEL 2026, aimed at banks, hospitals and telcos.

All conversation data is synthetic. Never type real card or ID numbers into it.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q
python -m dragonscope.data --out data
python -m dragonscope.train --data data --out runs/main
```

Training resumes from `latest.pt` if you rerun the same command. On a CPU it is slow, so use [notebooks/train_gpu.ipynb](notebooks/train_gpu.ipynb) on Colab or Kaggle.

## Run the demo

```bash
python -m dragonscope.server
```

Then open http://127.0.0.1:8000.

## Layout

| Path | Purpose |
|---|---|
| `dragonscope/engine.py` | Stateful inference, the memory matrix `rho`, and Forget |
| `dragonscope/data.py` | Synthetic conversation generator |
| `dragonscope/detect.py` | Card and NIC detection in statements and replies |
| `dragonscope/session.py` | Guided chat, flagged values and Forget for one session |
| `dragonscope/server.py` | FastAPI and WebSocket server, using `docs/PROTOCOL.md` |
| `dragonscope/activity.py` | Records which neurons really fire for each byte |
| `dragonscope/static/` | The exhibit page and 3D brain, with three.js bundled locally |
| `scripts/booth_check.py` | Runs the booth flow over a real WebSocket |
| `dragonscope/train.py` | Resumable training and exact-recall evaluation |
| `dragonscope/vendor/bdh/` | Unmodified upstream BDH, with its licence |
| `docs/PLAN.md` | Schedule, pass marks and fallbacks |
| `docs/STATUS.md` | Current state and next steps |

## Licence notice

`dragonscope/static/vendor/three.min.js` is three.js 0.160.0, under the licence in [three-LICENSE.txt](dragonscope/static/vendor/three-LICENSE.txt).

`dragonscope/vendor/bdh/` is from [pathwaycom/bdh](https://github.com/pathwaycom/bdh) at commit `2b0d7a45b058d4309c84a10e0768d541fe18bdc2`. It is copyright 2025 Pathway Technology, Inc., under the MIT licence in [its LICENSE.md](dragonscope/vendor/bdh/LICENSE.md).
