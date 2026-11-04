# Split-brain gateway (prototype)

The gateway lets an institution use an outside model, such as Claude, on confidential cases while identifiers stay on its own machine. A small BDH model holds the case in its memory. A de-identifier replaces identifiers with random codes, a leak gate checks everything before it leaves, and a re-linker puts the real values back locally. A named person reviews every draft.

**Status:** a student prototype on `main`, tested on synthetic data only. It is not part of the INFOTEL booth demo, which stays frozen on `infotel-2026`. The legal notes below are a good-faith reading of published rules, not legal advice, and no lawyer has reviewed them.

## Run it

```bash
python -m dragonscope.gateway                        # offline: nothing leaves the laptop
python -m dragonscope.gateway --connector anthropic  # sends cleared text to Claude
```

Open http://127.0.0.1:8200, load a sample note, then choose **Check and send**. The console shows:

- what was sent, with the codes highlighted;
- what was withheld;
- each gate layer's result;
- the draft with real values restored;
- buttons to approve or reject the draft, forget one fact, or erase the whole case.

The `anthropic` connector needs `pip install dragonscope[gateway-api]` and an API key. It uses `claude-opus-5-5`, with the API's server-side fallback, so a declined request can be retried on another Claude model. That other model receives the same cleared text.

Other options:
- `--block-on-memory` lets the experimental memory check block requests, not only report.
- `--audit-file audit.jsonl` also writes the audit log to a file.
- `--untrained` runs with random weights.

From Python:

```python
from dragonscope.gateway import Gateway
gateway = Gateway(model)
gateway.open_case("case-1", record_text)
draft = gateway.submit("case-1", "Write a discharge summary.", "Dr A, attending")
gateway.decide(draft.id, "Dr B, consultant", approve=True)
gateway.erase_case("case-1")
```

## What happens to a request

1. **Load.** The case is read into its own BDH memory, in one batched pass. That memory is the only copy of the case text.
2. **Register.** Every finding is recorded as a protected item, by kind and position.
3. **Replace.**
   - Direct identifiers become random codes such as `PERSON_QXKT`: names, card numbers, NICs, phone numbers, emails, record numbers, long numbers and districts.
   - Dates become their year, and ages over 89 become `90+`.
   - The same value gets the same code within one request, so "Mr Perera" gets the same code as "Nimal Perera".
4. **Gate.** Four layers check the outgoing text, including the fixed system prompt. Any flag blocks the request, and nothing is sent.
5. **Send.** Only the cleared text goes to the connector.
6. **Re-link.** Codes are swapped back to real values. The draft is flagged if the answer has an identifier the model wrote itself, or a code the request never issued.
7. **Review.** A named reviewer approves or rejects the draft.
8. **Log.** The audit log records the request. The code map is destroyed whatever happens.

Codes are random letters from Python's `secrets` module, so they are never derived from the data. Each map is used for one request, and the way back never leaves the process. That follows HIPAA's rule for re-identification codes.

## The leak gate

| Layer | What it checks |
|---|---|
| 1. Format | Anything shaped like a card number, NIC or phone number; any email; any run of 5 or more digits; any date finer than a year; any age over 89. These rules are broader than the de-identifier's, so the two are not the same check. |
| 2. Lexicon | Listed names and places; names after a title or label, such as "Dr" or "Name:"; any protected value from the case appearing again. |
| 3. Combination | Scores quasi-identifiers: a revealing occupation or a rare condition scores 2, and an exact age scores 1. At 2 or more, occupations and conditions are generalised first ("Minister of Fisheries" becomes "public official"), then ages become decades. After two rounds, a person decides. |
| 4. Memory (experimental) | BDH reads the case and then the outgoing text. For each protected item, the score is the attention the outgoing text pays to that item's positions in memory, divided by the item's share of the case. In the engine's terms, this is the item's share of `rho @ QR_t`. It reports only, unless `--block-on-memory` is set. |

## Measured

Results are from `python scripts/gateway_eval.py` with the trained demo model, the offline connector and synthetic data. They are in `docs/gateway/results.json`.

| Measure | Result |
|---|---|
| Labelled identifiers that reached the outgoing text | 50 of 1,665 (3.0%) |
| Names, cued ("Patient:", "Mr") or in the word lists | 0 leaked |
| Names not in the lists and with no cue | 50 of 50 leaked |
| Cards, NICs, phones, record numbers, districts, ages over 89 | 0 leaked |
| Clean requests blocked | 0 of 100 |
| Time per request, model loaded | 0.9 s mean, 3.5 s worst |

**The rules miss what they were not written for.** Every leak was a name that is not in the word lists and had no cue. A real deployment needs a trained name detector here (the design's layer 2), and Expert Determination to measure what is left.

**The memory check did not beat the rules.** On 100 paraphrases of a protected occupation, such as "The patient leads the fisheries ministry", and 100 clean control sentences:

- layers 1 to 3 caught no paraphrase, as expected, because none uses a listed word;
- the memory check flagged neither paraphrases nor controls at its threshold of 3;
- its mean score was 1.13 for paraphrases and 1.10 for controls;
- a paraphrase outscored a control in 66% of pairs, where 50% would be chance.

So it carries a weak signal, far too weak to block on. This is the demo model, trained only on short bank chats, so the result says little about a model trained on records. Until a test shows otherwise, layer 4 stays report-only, and layers 1 to 3 do the protecting.

## What is built and what is not

| Part of the design | State |
|---|---|
| Record adapter | Not built. Cases are pasted or loaded from synthetic samples. |
| Private brain (BDH) | Built: one memory per case, built in one batched pass. Limited to 2,000 bytes. |
| De-identifier | Built from rules and word lists. **Gap:** no trained name detector, and no Sinhala or Tamil names. |
| Placeholder issuer and re-linker | Built. |
| Leak gate, layers 1 to 3 | Built. |
| Leak gate, layer 4 | Built as an experiment. It reports only, because it is unproven (see above). |
| Frontier connector | Built: an offline stub, and Claude through the official SDK, at one fixed endpoint, with no metadata. |
| Review console | Built, on 127.0.0.1 only. **Gap:** no sign-in, roles or case assignment. Anyone who can reach the port can use it. |
| Audit log | Built, hash-chained. **Gap:** in memory or a plain file, not encrypted. |
| Forget one fact | Built: `rebuild`, with the residual measured. It was 0 in tests. |
| Erase a person | Built: the case memory, its drafts and the text sent for it. |
| Encryption at rest, key management | Not built. Everything is in RAM. |

## The audit log

Entries hold no protected value and no hash of one. A withheld item is recorded as its kind and, for numbers, the last 4 digits, as the booth demo's deletion log does. The design document proposed a salted hash; that was dropped, because a hash of a short value such as a NIC can be reversed by trying every value.

Each entry's hash covers the entry and the previous hash, so changing or removing an entry breaks the chain, and the console shows whether the chain is intact. The cleared text that was sent is kept beside the chain, not in it. Erasing a case therefore removes that text without breaking the chain. The cleared text holds no identifiers when the gate works, and erasing removes it from the log either way.

## Legal alignment (not legal advice)

The design rests on one idea: vendor obligations attach to identifiable data. How far that carries depends on the jurisdiction.

| Regime | If only de-identified text leaves | What still applies |
|---|---|---|
| US HIPAA | HHS says a cloud provider that receives only de-identified data is not a business associate. The condition is that de-identification meets Safe Harbor or Expert Determination, which the 3% leak rate above shows rules alone do not. | A Security Rule risk analysis for the on-premises system. Expert Determination is the safer route. |
| US GLBA | The AI vendor likely falls outside the service-provider rules if it never receives customer information. | The bank's own Safeguards Rule program, and third-party oversight of any outside AI service. |
| EU GDPR | Codes are pseudonymisation, so GDPR normally still applies to the vendor. *EDPS v SRB* (2025) opened an argument for recipients with no means to re-identify, but do not rely on it without advice. | An Article 28 processor contract, a transfer mechanism, and a DPIA. |
| Sri Lanka PDPA | Unclear. There is no guidance yet on de-identified data. Parts I and III apply from 1 January 2027. | Treat the vendor as receiving personal data. Section 26 governs transfers. CBSL Direction No. 1 of 2026 may treat an outside AI service as outsourcing. |

## Rules this follows

- Nothing logs a protected value, and errors never repeat submitted text.
- Erasing a case leaves no copy in the case memory, drafts, console state or audit file. A test checks each one.
- It runs offline by default. The console page loads nothing from outside.
- Engine and vendored BDH code are unchanged.
