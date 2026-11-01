# WebSocket protocol

The dashboard talks to `python -m dragonscope.server` at `ws://127.0.0.1:8000/ws`. Messages are JSON. There is one shared session, and every connected client receives each update. A connection from another web origin is refused.

**Replace your whole view with each `state` message, and keep no earlier copy.** This is how a forgotten value leaves the chat display: `state` is the only message that carries chat text, so do not store it in history, logs or `localStorage`.

## Client to server

Each accepted message is followed by a `state` message.

| Message | Effect |
|---|---|
| `{"type": "say", "field": "card", "value": "4111 1111 1111 1111"}` | States a value in its trained wording, and the model replies. |
| `{"type": "ask", "field": "card"}` | Asks for a field in its trained wording. Use this for "ask it again". |
| `{"type": "forget", "finding": 3}` | Erases that finding and every linked copy. A `forget_result` arrives before the `state`. |
| `{"type": "reset"}` | Clears the session for the next visitor, including the deletion log. |

`field` is `card`, `nic`, `name` or `branch`. There is no free-text message, because the model only understands the wordings it was trained on.

## Server to client

### `hello`

Sent once on connecting. It has a `state` (below) and a `guide` holding the only values the chat can offer:

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
    "neurons": [
      {"layer": 0, "head": 2, "neuron": 317, "strength": 9.81, "flagged_share": 0.42}
    ]
  }
}
```

- **`segments`:** join `text` to get a turn. A segment with `finding` is flagged, so draw the alert from it. After Forget, the value's segment becomes `{"text": "", "forgotten": true}`.
- **`findings`:** every flagged span. `kind` is `card`, `nic` or `number`. `source` is `user` or `reply`. A reply's `luhn_valid` is `false` when it is a garbled echo of a card.
- **`stated`:** fields given this session. A field can be given only once per session, even after Forget, so disable those inputs until reset.
- **`last_answer`:** set after an `ask`, and `refused` is true when the reply is exactly the refusal. It is `null` after anything else.
- **`memory.neurons`:** at most 150 neurons for the graph. This is the strongest set, plus the 30 holding most of any flagged data. `flagged_share` is that data's strength relative to the neuron's total, from 0 to 1. Call these neurons in a compressed form of the synaptic state, not individual synapses.
- **`position` and `position_limit`:** Forget does not free positions. When an `ask` or `say` would pass the limit, the server sends `session_full`. A visitor can state all four fields, ask once, Forget, ask again, and ask about at least one other field.

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

Replies arrive as a whole, not byte by byte. On the demo laptop on battery, a reply takes under a second and Forget takes about 0.8 s, so show a short "erasing" state.
