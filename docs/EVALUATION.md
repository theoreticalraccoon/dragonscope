# Forget evaluation

How well does Forget work on the bank demo model? This page has the 500-case evaluation. The forget-versus-rebuild benchmark across conversation lengths is in the [README](../README.md#forget-versus-rebuild), with raw numbers in [benchmark/](benchmark/).

## Setup

- Model: `models/best_model.pt`, 1.6M parameters, 4 layers of 4,096 neurons, trained on synthetic bank conversations of up to 512 bytes.
- 500 conversations with new random values, not seen in training. Half forget a card number and half a NIC.
- Greedy generation. Every copy of the value is forgotten, including any the model had already repeated.
- Raw results: [forget_eval_500.json](forget_eval_500.json). Reproduce with `python -m dragonscope.evaluate_forget`.

## Results

| Check | Target | Before Forget | `subtract` | `rebuild` |
|---|---|---|---|---|
| Exact recall, trained question | 1% or less | 100% | 0.0% | 0.0% |
| Exact recall, reply forced to start "Your card number is " | 1% or less | 100% | 0.0% | 0.0% |
| Digits matching on the forced reply (control: an unrelated number) | About 10%, chance | 100% | 9.9% (10.1%) | 10.1% (10.8%) |
| Longest shared run of digits, worst case (control) | Same as control | 16 | 4 (4) | 4 (4) |
| Other fields kept | Within 5 points | 100% | 100% | 100% |
| "I don't have that on record." reply | 90% or more | n/a | 100% | 100% |
| Same, after the model had already repeated the value | 90% or more | n/a | 100% | 100% |
| Distance from a memory that never saw the value | Record it | n/a | 3.8% (1.8 to 5.4%) | 0 |

The forced-reply checks matter most. They make the model start its answer as if it remembered, so any trace it could use would show up as digits above chance. None did.

## Limits

- **Forget every copy.** Forgetting only the original statement, after the model had repeated the number, gave `4711 1111 1111 5111` for the real `4111 1111 1111 1111`. The demo forgets every copy, including number-like spans in its own replies.
- **`subtract` leaves a residual** of a few percent from the second layer on, even though no probe could use it. Only `rebuild` matches a memory that never saw the value.
- **Reworded questions are a weak probe.** Before Forget, unseen wordings recalled only 37.6% of values, so the demo uses trained wordings.
- **Names and branches.** The model memorised the training lists of names and branches and cannot copy new ones, so the demo centres on card and NIC numbers.
- **Not a security proof.** This is one small model, synthetic data and these probes. The weights never change, so this is not machine unlearning.
