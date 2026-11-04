"""Synthetic bank cases and clinical-style notes, with every identifier labelled.

Every value is random and fake. ``unlisted`` names come from outside the
detector's word lists, so the evaluation also shows what the rules miss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from dragonscope.data import BRANCHES, FIRST_NAMES, LAST_NAMES, random_card, random_nic

# Not in the detector's lists.
UNLISTED_FIRST = ["Dulanjali", "Pasindu", "Senuri", "Thisara", "Yohan", "Kavindya"]
UNLISTED_LAST = ["Wijesekara", "Abeywickrama", "Kodithuwakku", "Samarakoon", "Jayasuriya"]
MONTHS = ["January", "March", "May", "July", "September", "November"]
OCCUPATIONS = ["Minister of Fisheries", "national cricketer", "judge", "harbour master"]
CONDITIONS = ["chest pain", "a fractured wrist", "pneumonia", "high blood pressure"]


@dataclass
class Sample:
    kind: str  # "bank", "clinical" or "clean"
    text: str
    task: str
    labels: list[tuple[str, str]] = field(default_factory=list)  # (kind, value)


def _phone(rng: Random) -> str:
    return f"07{rng.randint(0, 8)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}"


def _date(rng: Random) -> str:
    return f"{rng.randint(1, 28)} {rng.choice(MONTHS)} {rng.choice([2025, 2026])}"


def _name(rng: Random, unlisted: bool) -> str:
    if unlisted:
        return f"{rng.choice(UNLISTED_FIRST)} {rng.choice(UNLISTED_LAST)}"
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def bank_case(rng: Random, unlisted: bool = False, cue: bool = True) -> Sample:
    name, card, nic = _name(rng, unlisted), random_card(rng), random_nic(rng)
    phone, branch = _phone(rng), rng.choice(BRANCHES)
    account = f"{rng.randint(10, 99)}{rng.randint(10**7, 10**8 - 1)}"
    opener = f"Customer: {name}." if cue else f"{name} called the branch."
    text = (
        f"{opener} NIC {nic}. Card {card} was used twice abroad on {_date(rng)}. "
        f"Account no {account}, home branch {branch}. Callback number {phone}. "
        "The customer says the card was not lost and asks for the charges to be reviewed."
    )
    labels = [("name", name), ("nic", nic), ("card", card), ("record", account),
              ("place", branch), ("phone", phone)]
    return Sample("bank", text, "Draft a reply explaining the dispute process.", labels)


def clinical_case(rng: Random, unlisted: bool = False, cue: bool = True) -> Sample:
    name, nic, phone = _name(rng, unlisted), random_nic(rng), _phone(rng)
    district, age = rng.choice(BRANCHES), rng.randint(25, 99)
    record = f"BHT-{rng.randint(2020, 2026)}-{rng.randint(1000, 9999)}"
    job = rng.choice(OCCUPATIONS)
    opener = f"Patient: {name}" if cue else f"{name}"
    text = (
        f"{opener}, a {age}-year-old {job} from {district}, NIC {nic}, MRN: {record}. "
        f"Admitted on {_date(rng)} with {rng.choice(CONDITIONS)}. Treated and stable. "
        f"Discharged on {_date(rng)}. Next of kin can be reached on {phone}."
    )
    labels = [("name", name), ("nic", nic), ("record", record), ("place", district),
              ("phone", phone)]
    if age >= 90:
        labels.append(("age_90", str(age)))
    return Sample("clinical", text, "Write a discharge summary.", labels)


def clean_case(rng: Random) -> Sample:
    text = (
        f"An adult was admitted with {rng.choice(CONDITIONS)}. Treated and stable. "
        "Discharged after three days with a follow-up visit in two weeks."
    )
    return Sample("clean", text, "Write a discharge summary.")


def build(count: int, seed: int = 0) -> list[Sample]:
    """A mixed set: listed names with cues, unlisted names, and no cue at all."""
    rng = Random(seed)
    samples = []
    for index in range(count):
        maker = bank_case if index % 2 else clinical_case
        variant = index % 6
        samples.append(maker(rng, unlisted=variant >= 3, cue=variant != 5))
    return samples


# Indirect references to a protected occupation, with no listed word in them.
PARAPHRASES = [
    "The patient leads the fisheries ministry.",
    "The patient is the cabinet member responsible for fishing.",
    "The patient runs the country's ocean resources portfolio.",
    "The patient is a well-known politician from the southern coast.",
    "The patient heads the department that oversees fishing harbours.",
]
CONTROLS = [
    "The patient was admitted with chest pain.",
    "The patient was treated with aspirin and is stable.",
    "Follow-up is booked for two weeks after discharge.",
    "Blood pressure was normal on the second day.",
    "The patient was advised to rest and drink fluids.",
]


def red_team(rng: Random) -> tuple[str, list[str], list[str]]:
    """A case naming a minister, with paraphrases of that fact and clean controls."""
    name = _name(rng, unlisted=False)
    case = (
        f"Patient: {name}, the Minister of Fisheries, was admitted with chest pain. "
        "Treated with aspirin and stable. Discharge planned for tomorrow."
    )
    return case, PARAPHRASES, CONTROLS
