# WebSocket protocol

The dashboard in `dragonscope/static/` talks to `python -m dragonscope.server` at `ws://127.0.0.1:8000/ws`. Messages are JSON. There is one shared session, and every connected client receives each update. A connection from another web origin is refused.

**Replace your whole view with each `state` message, and keep no earlier copy.** This is how a forgotten value leaves the chat display. `state` carries the chat text, and each `frame` carries only the one character being computed. Do not store either in history, logs or `localStorage`, and drop frames when the next `state` arrives.

## Client to server

Each accepted message is followed by a `state` message. A `say` or `ask` sends `frame` messages first.

| Message | Effect |
|---|---|
| `{"type": "say", "field": "card", "value": "4111 1111 1111 1111"}` | States a value in its trained wording, and the model replies. |
| `{"type": "ask", "field": "card"}` | Asks for a field in its trained wording. Use this for "ask it again". |
| `{"type": "forget", "finding": 3}` | Erases that finding and every linked copy. A `forget_result` arrives before the `state`. |
| `{"type": "speed", "full": true}` | Drops the pacing that holds each byte long enough to watch, or restores it. |
| `{"type": "reset"}` | Clears the session for the next visitor, including the deletion log. |

`field` is `card`, `nic`, `name` or `branch`. There is no free-text message, because the model only understands the wordings it was trained on.

## Server to client

### `hello`

Sent once on connecting. It has a `state` (below), a `model` with the real model's size (`neurons`, `heads`, `per_head`, `rounds`, `parameters`) and each neuron's place within its head (`u`, `v`, from 0 to 1, from what it responds to), and a `guide` holding the only values the chat can offer:

```json
{
  "type": "hello",
  "guide": {
    "fields": ["card", "nic", "name", "branch"],
    "statements": {"card": "My card number is {}."},
    "questions": {"card": "What is my card number?"},
    "first_names": ["Nimal"],
    "last_names": ["Perera"],
    "branches": ["Colombo"],
    "refusal": "I don't have that on record."
  },
  "state": {}
}
```

The lists are shortened here. `statements` and `questions` have an entry for each field. Name and branch must come from these lists, and a name is `"First Last"`.

### `state`

```json
{
  "type": "state",
  "turns": [
    {
      "role": "user",
      "segments": [
        {"text": "My card number is "},
        {"text": "4111 1111 1111 1111", "finding": 1},
        {"text": "."}
      ]
    },
    {"role": "assistant", "segments": [{"text": "Noted."}]}
  ],
  "findings": [
    {"id": 1, "kind": "card", "source": "user", "turn": 0, "luhn_valid": true, "last4": "1111"}
  ],
  "stated": ["card"],
  "last_answer": {"field": "card", "refused": false},
  "deletion_log": [{"type": "card", "last4": "1111", "time": "2026-11-06T04:30:00+00:00"}],
  "memory": {
    "bytes_stored": 62,
    "ledger_entries": 62,
    "position": 83,
    "position_limit": 400,
    "rho_norm": 412.7,
    "strength": [12, 0, 47],
    "flagged": [0, 0, 63]
  }
}
```

- **`segments`:** join `text` to get a turn. A segment with `finding` is flagged, so draw the alert from it. After Forget, the value's segment becomes `{"text": "", "forgotten": true}`.
- **`findings`:** every flagged span. `kind` is `card`, `nic` or `number`. `source` is `user` or `reply`. A reply's `luhn_valid` is `false` when it is a garbled echo of a card.
- **`stated`:** fields given this session. A field can be given only once per session, even after Forget, so disable those inputs until reset.
- **`last_answer`:** set after an `ask`, and `refused` is true when the reply is exactly the refusal. It is `null` after anything else.
- **`memory.strength` and `memory.flagged`:** one integer per neuron, from 0 to 100, combined across rounds. `strength` is what the neuron holds relative to the strongest. `flagged` is the share of that from flagged data. This memory is a compressed form of the synaptic state, not individual synapses.
- **`position` and `position_limit`:** Forget does not free positions. When an `ask` or `say` would pass the limit, the server sends `session_full`. A visitor can state all four fields, ask once, Forget, ask again, and ask about at least one other field.

### `frame`

Sent live for each byte, as the model computes it, before the `state` of a `say` or `ask`.

```json
{
  "type": "frame",
  "phase": "write",
  "char": "4",
  "question": true,
  "x": ["base64", "base64", "base64", "base64"],
  "change": ["base64", "base64", "base64", "base64"],
  "counts": [881, 874, 829, 866],
  "passed": [129, 118, 68, 70],
  "links": [[[2176, 3049]], [[164, 2210]], []],
  "depends": 0.98,
  "ms": 27.1,
  "show_ms": 110
}
```

- **`char`:** the character being read or written, empty for chat markers and newlines.
- **`x`:** per round, every neuron's activation as one byte, from 0 to 255 on a fixed scale of 7. There are 4,096 bytes, indexed `head * 1024 + neuron`.
- **`change`:** per round, on the same scale, how much each neuron's firing differs from a copy of memory that never read the flagged bytes. It is empty when nothing is flagged. The first round is always 0, as it depends only on the byte.
- **`counts` and `passed`:** per round, how many neurons fired, and how many also passed a signal on.
- **`links`:** for each pair of consecutive rounds, the `[from, to]` learned connections most used: `decoder[from] . encoder[to]`, weighted by the signal `from` passed and how strongly `to` then fired.
- **`depends`:** for written bytes, how much of the probability of writing this byte disappears in that copy.
- **`ms` and `show_ms`:** the real time to compute the byte, comparison included, and the time it is held for watching.

The model has 4,096 neurons. Its encoder is shared by all layers, so each byte passes through the same neurons in 4 rounds.

Frames hold the byte itself, so they are sensitive. The server does not keep them. **Drop them when the next `state` arrives.**

### `forget_result`

```json
{
  "type": "forget_result",
  "log_entry": {"type": "card", "last4": "1111", "time": "2026-11-06T04:30:00+00:00"},
  "copies_erased": 2,
  "bytes_erased": 38,
  "memory_change": 0.21,
  "residual_after_subtract": 0.048,
  "residual_after_rebuild": 0.0,
  "subtract_ms": 41.0,
  "rebuild_ms": 310.0,
  "checks": {"value_in_memory_bytes": false, "ledger_entries_left": 0, "pending_logits": false}
}
```

The numbers are measured on each Forget. `memory_change` is the relative change in `rho`. `residual_after_subtract` shows what `subtract` left behind, so do not present that step as complete erasure.

### `error`

Sent only to the client that caused it, and it never repeats the submitted value.

```json
{"type": "error", "code": "invalid_value", "message": "That is not a valid card number."}
```

Codes: `invalid_value`, `already_stated`, `session_full`, `unknown_field`, `unknown_finding`, `unknown_type`, `bad_json`, `too_long`, `internal`.

## Timing

A byte took about 45 to 85 ms on the demo laptop on battery with the page open, comparison included, and the page shows the real figure. The server holds each byte for 75 ms when reading and 110 ms when writing, unless `full_speed` is on, so the model is running as the page shows it. Forget takes about 0.4 to 0.8 s.
