"""Generate synthetic bank conversations. Every value here is random and fake.

Each field is given, absent, blank or masked. The last three all teach the reply
``I don't have that on record.``, so after Forget the model should say that
rather than invent digits. Blank and masked mimic what is left behind after
Forget: the sentence survives but the value is gone.

Run: ``python -m dragonscope.data --out data``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from random import Random

import numpy as np

REFUSAL = "I don't have that on record."
MAX_BYTES = 512
FIELDS = ("card", "nic", "name", "branch")
STATES = ("given", "absent", "blank", "masked")
STATE_WEIGHTS = (0.70, 0.12, 0.09, 0.09)
KIND_RECALL = 0
KIND_REFUSAL = 1

FIRST_NAMES = """Nimal Kamal Sunil Amara Saman Chamari Dilani Ruwan Tharindu Nadeesha
Kasun Sanduni Ishara Malith Priya Lakshan Hiruni Dinesh Anjali Roshan Thilini
Mohamed Fathima Rizwan Ayesha Kumar Selvi Arjun Meena Harsha Gayani Janith""".split()
LAST_NAMES = """Perera Silva Fernando Jayawardena Bandara Dissanayake Wickramasinghe
Gunawardena Rajapaksa Herath Ranasinghe Senanayake Mendis Weerasinghe Karunaratne
Nazeer Hassan Sivakumar Rajan Pillai Amarasinghe Liyanage Ekanayake""".split()
BRANCHES = """Colombo Kandy Galle Jaffna Negombo Matara Kurunegala Anuradhapura
Trincomalee Batticaloa Ratnapura Badulla Matale Kalutara Gampaha Chilaw Hambantota
Vavuniya Kegalle Polonnaruwa Moratuwa Dehiwala Panadura""".split()
# These never appear in training, so validation tests copying rather than memory.
HOLDOUT_FIRST = 5
HOLDOUT_LAST = 4
HOLDOUT_BRANCHES = 4

STATEMENTS = {
    "card": ("My card number is {}.", "Hi, my card is {}.", "Here is my card number: {}."),
    "nic": ("My NIC is {}.", "My NIC number is {}.", "Here is my NIC: {}."),
    "name": ("My name is {}.", "The account name is {}.", "This is {}."),
    "branch": ("My branch is {}.", "I bank at the {} branch.", "My home branch is {}."),
}
QUESTIONS = {
    "card": ("What is my card number?", "Can you tell me my card number?", "Remind me of my card."),
    "nic": ("What is my NIC?", "Can you tell me my NIC number?", "Remind me of my NIC."),
    "name": ("What is my name?", "What is the account name?", "Remind me of my name."),
    "branch": ("What is my branch?", "Which branch do I bank at?", "Remind me of my branch."),
}
ANSWERS = {
    "card": "Your card number is {}.",
    "nic": "Your NIC is {}.",
    "name": "Your name is {}.",
    "branch": "Your branch is {}.",
}
ACKNOWLEDGEMENTS = ("Noted.", "Thanks, noted.", "Got it.", "Thank you.")
SMALL_TALK = (
    ("What are your opening hours?", "We are open 9 to 5 on weekdays."),
    ("Do you have a mobile app?", "Yes, our app is free to download."),
    ("Hello, can you help me?", "Yes, how can I help?"),
)


def luhn_check_digit(digits: str) -> str:
    total = 0
    for index, character in enumerate(reversed(digits)):
        digit = int(character)
        if index % 2 == 0:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return str((10 - total % 10) % 10)


def luhn_valid(number: str) -> bool:
    digits = "".join(c for c in number if c.isdigit())
    return luhn_check_digit(digits[:-1]) == digits[-1]


def random_card(rng: Random) -> str:
    body = rng.choice("45") + "".join(rng.choice("0123456789") for _ in range(14))
    number = body + luhn_check_digit(body)
    groups = [number[i : i + 4] for i in range(0, 16, 4)]
    separator = rng.choices((" ", "-", ""), weights=(6, 2, 2))[0]
    return separator.join(groups)


def random_nic(rng: Random) -> str:
    if rng.random() < 0.5:
        return "".join(rng.choice("0123456789") for _ in range(9)) + rng.choice("VX")
    return "".join(rng.choice("0123456789") for _ in range(12))


def split_names(split: str) -> tuple[list[str], list[str], list[str]]:
    if split == "train":
        return (
            FIRST_NAMES[:-HOLDOUT_FIRST],
            LAST_NAMES[:-HOLDOUT_LAST],
            BRANCHES[:-HOLDOUT_BRANCHES],
        )
    return (
        FIRST_NAMES[-HOLDOUT_FIRST:],
        LAST_NAMES[-HOLDOUT_LAST:],
        BRANCHES[-HOLDOUT_BRANCHES:],
    )


def mask(value: str) -> str:
    return "".join("#" if c.isalnum() else c for c in value)


@dataclass(frozen=True)
class Answer:
    start: int
    end: int
    kind: int
    field: int
    state: int


def build_conversation(rng: Random, split: str) -> tuple[str, list[Answer]]:
    first_names, last_names, branches = split_names(split)
    values = {
        "card": random_card(rng),
        "nic": random_nic(rng),
        "name": f"{rng.choice(first_names)} {rng.choice(last_names)}",
        "branch": rng.choice(branches),
    }
    states = {
        field: rng.choices(STATES, weights=STATE_WEIGHTS)[0] for field in FIELDS
    }

    order = list(FIELDS)
    rng.shuffle(order)
    text = ""
    small_talk_after = rng.randrange(len(order)) if rng.random() < 0.3 else None
    for position, field in enumerate(order):
        if states[field] != "absent":
            shown = {
                "given": values[field],
                "blank": "",
                "masked": mask(values[field]),
            }[states[field]]
            statement = rng.choice(STATEMENTS[field]).format(shown)
            text += f"U: {statement}\nA: {rng.choice(ACKNOWLEDGEMENTS)}\n"
        if position == small_talk_after:
            question, reply = rng.choice(SMALL_TALK)
            text += f"U: {question}\nA: {reply}\n"

    answers = []
    for field in rng.sample(FIELDS, k=rng.randint(1, 2)):
        text += f"U: {rng.choice(QUESTIONS[field])}\nA: "
        if states[field] == "given":
            reply = ANSWERS[field].format(values[field])
            kind = KIND_RECALL
        else:
            reply = REFUSAL
            kind = KIND_REFUSAL
        start = len(text)
        text += reply + "\n"
        answers.append(
            Answer(
                start=start,
                end=len(text),
                kind=kind,
                field=FIELDS.index(field),
                state=STATES.index(states[field]),
            )
        )
    return text, answers


def build_dataset(count: int, seed: int, split: str) -> dict[str, np.ndarray]:
    rng = Random(seed)
    chunks: list[bytes] = []
    offsets = [0]
    rows: list[tuple[int, int, int, int, int, int]] = []
    for conversation in range(count):
        text, answers = build_conversation(rng, split)
        encoded = text.encode("ascii")
        if len(encoded) > MAX_BYTES:
            raise ValueError(f"conversation is {len(encoded)} bytes, over {MAX_BYTES}")
        chunks.append(encoded)
        offsets.append(offsets[-1] + len(encoded))
        rows.extend(
            (conversation, a.start, a.end, a.kind, a.field, a.state) for a in answers
        )
    return {
        "bytes": np.frombuffer(b"".join(chunks), dtype=np.uint8),
        "offsets": np.array(offsets, dtype=np.int64),
        "answers": np.array(rows, dtype=np.int32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--train", type=int, default=60_000)
    parser.add_argument("--val", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for split, count, seed in (
        ("train", args.train, args.seed),
        ("val", args.val, args.seed + 1),
    ):
        dataset = build_dataset(count, seed, split)
        np.savez(args.out / f"{split}.npz", **dataset)
        lengths = np.diff(dataset["offsets"])
        print(
            f"{split}: {count} conversations, {len(dataset['bytes'])} bytes, "
            f"length mean {lengths.mean():.0f} max {lengths.max()}, "
            f"{len(dataset['answers'])} answers"
        )


if __name__ == "__main__":
    main()
