"""Find identifiers and quasi-identifiers in free text.

Three kinds of finding:

* **Direct identifiers** (names, card numbers, NICs, phones, emails, record
  numbers, long numbers, places) are replaced by placeholders.
* **Dates and ages over 89** are generalised, as HIPAA Safe Harbor asks: a date
  becomes its year and an age over 89 becomes ``90+``.
* **Quasi-identifiers** (exact ages, revealing occupations, rare conditions) stay
  unless the leak gate's combination check decides they single someone out.

Rules and word lists catch what they were written for. Names that are not in the
lists and appear without a cue such as "Mr" or "Name:" are missed; this module
stands in for a trained name detector, which the prototype does not have.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dragonscope.data import BRANCHES, FIRST_NAMES, LAST_NAMES, luhn_valid
from dragonscope.detect import CARD, NIC

DIRECT = ("name", "card", "nic", "phone", "email", "record", "number", "place")
GENERALISED = ("date", "age_90")
QUASI = ("age", "occupation", "condition")
# Kinds whose last 4 digits may be logged, as the project's deletion logs do.
DIGIT_KINDS = ("card", "nic", "phone", "record", "number")

PHONE = re.compile(r"(?<![\d+])(?:\+94[\s-]?|0)\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
RECORD = re.compile(
    r"\b(?:MRN|BHT|[Aa]ccount|[Rr]ecord|[Pp]olicy|[Rr]eference|[Pp]assport)"
    r"(?:\s+(?:[Nn]o\.?|[Nn]umber|ID))?\s*[:#]?\s*"
    r"(?P<value>(?=[A-Z-]*\d)[A-Z0-9][A-Z0-9-]{4,})\b"
)
LONG_NUMBER = re.compile(r"(?<![\w+])\d[\d -]{3,}\d(?![\w])")
MIN_NUMBER_DIGITS = 5

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
    "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
DATES = [
    re.compile(r"(?P<on>\bon\s+)?\b(?P<year>(?:19|20)\d{2})-\d{2}-\d{2}\b"),
    re.compile(r"(?P<on>\bon\s+)?\b\d{1,2}[/.-]\d{1,2}[/.-](?P<year>(?:19|20)\d{2})\b"),
    re.compile(
        rf"(?P<on>\bon\s+)?\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{MONTHS})\.?,?\s+"
        r"(?P<year>(?:19|20)\d{2})\b"
    ),
    re.compile(
        rf"(?P<on>\bon\s+)?\b(?:{MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+"
        r"(?P<year>(?:19|20)\d{2})\b"
    ),
    re.compile(rf"(?P<on>\bin\s+)?\b(?:{MONTHS})\.?\s+(?P<year>(?:19|20)\d{{2}})\b"),
]
AGES = [
    re.compile(r"\b(?P<age>\d{1,3})(?=[- ]years?[- ]old\b|[- ]year-old\b|\s*(?:yrs?|y/o)\b)"),
    re.compile(r"\b[Aa]ged?\s*:?\s*(?P<age>\d{1,3})\b"),
]

TITLE_NAME = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Rev|Hon)\.?\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
)
LABEL_NAME = re.compile(
    r"\b(?:[Nn]ame|[Pp]atient|[Cc]ustomer|[Cc]lient|[Aa]ccount holder|[Nn]ext of kin)\s*[:\-]\s*"
    r"(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
)
CUE_NAME = re.compile(
    r"\b(?:[Mm]y name is|[Tt]his is|named|called)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
)
_KNOWN_NAMES = sorted(set(FIRST_NAMES) | set(LAST_NAMES), key=len, reverse=True)
LEXICON_NAME = re.compile(
    r"\b(?:" + "|".join(_KNOWN_NAMES) + r")(?:\s+(?:" + "|".join(_KNOWN_NAMES) + r"))*\b"
)
PLACE = re.compile(r"\b(?:" + "|".join(sorted(BRANCHES, key=len, reverse=True)) + r")\b")

# Revealing occupations and what they become when generalised.
OCCUPATIONS = {
    "senior government official": "public sector employee",
    "member of parliament": "public official",
    "minister": "public official",
    "deputy minister": "public official",
    "ambassador": "public official",
    "governor": "public official",
    "mayor": "public official",
    "inspector general": "public official",
    "police chief": "public official",
    "judge": "legal professional",
    "magistrate": "legal professional",
    "bishop": "religious leader",
    "chief monk": "religious leader",
    "national cricketer": "athlete",
    "cricketer": "athlete",
    "olympian": "athlete",
    "news anchor": "media worker",
    "journalist": "media worker",
    "film actor": "media worker",
    "actress": "media worker",
    "dock worker": "manual worker",
    "harbour master": "port employee",
    "surgeon general": "senior clinician",
    "bank chairman": "business executive",
    "chief executive": "business executive",
}
# Rare conditions and their generalised form.
CONDITIONS = {
    "huntington's disease": "a rare neurological condition",
    "huntington disease": "a rare neurological condition",
    "creutzfeldt-jakob disease": "a rare neurological condition",
    "amyotrophic lateral sclerosis": "a rare neurological condition",
    "progeria": "a rare genetic condition",
    "fibrodysplasia ossificans progressiva": "a rare genetic condition",
    "wilson's disease": "a rare metabolic condition",
    "gaucher disease": "a rare metabolic condition",
    "pompe disease": "a rare metabolic condition",
    "kuru": "a rare neurological condition",
}
# Only the phrase ignores case, so "of Fisheries" extends it but "of the" does not.
OCCUPATION = re.compile(
    r"\b(?i:" + "|".join(re.escape(o) for o in sorted(OCCUPATIONS, key=len, reverse=True)) + r")"
    r"(?:\s+of\s+[A-Z][a-z]+(?:\s+(?:and\s+)?[A-Z][a-z]+)*)?\b"
)
CONDITION = re.compile(
    r"\b(?i:" + "|".join(re.escape(c) for c in sorted(CONDITIONS, key=len, reverse=True)) + r")"
)

# Earlier kinds win when two findings overlap.
PRIORITY = (
    "card", "nic", "email", "phone", "record", "date", "age_90", "age",
    "name", "place", "occupation", "condition", "number",
)


@dataclass(frozen=True)
class Identifier:
    kind: str
    start: int
    end: int
    text: str
    replacement: str | None = None  # For generalised kinds: what replaces the text.

    @property
    def last4(self) -> str | None:
        """The last 4 digits, for kinds whose deletion log may keep them."""
        if self.kind not in DIGIT_KINDS:
            return None
        digits = "".join(c for c in self.text if c.isdigit())
        return digits[-4:] or None


def _year(match: re.Match) -> str:
    year = match.group("year")
    return f"in {year}" if match.group("on") else year


def _candidates(text: str) -> list[Identifier]:
    found: list[Identifier] = []

    def add(kind: str, start: int, end: int, replacement: str | None = None) -> None:
        found.append(Identifier(kind, start, end, text[start:end], replacement))

    for match in CARD.finditer(text):
        if luhn_valid(match.group()):
            add("card", match.start(), match.end())
    for match in NIC.finditer(text):
        add("nic", match.start(), match.end())
    for match in EMAIL.finditer(text):
        add("email", match.start(), match.end())
    for match in PHONE.finditer(text):
        add("phone", match.start(), match.end())
    for match in RECORD.finditer(text):
        add("record", match.start("value"), match.end("value"))
    for pattern in DATES:
        for match in pattern.finditer(text):
            add("date", match.start(), match.end(), _year(match))
    for pattern in AGES:
        for match in pattern.finditer(text):
            age = int(match.group("age"))
            if age >= 90:
                add("age_90", match.start("age"), match.end("age"), "90+")
            elif age >= 1:
                add("age", match.start("age"), match.end("age"))
    for pattern in (TITLE_NAME, LABEL_NAME, CUE_NAME):
        for match in pattern.finditer(text):
            add("name", match.start("name"), match.end("name"))
    for match in LEXICON_NAME.finditer(text):
        add("name", match.start(), match.end())
    for match in PLACE.finditer(text):
        add("place", match.start(), match.end())
    for match in OCCUPATION.finditer(text):
        add("occupation", match.start(), match.end())
    for match in CONDITION.finditer(text):
        add("condition", match.start(), match.end())
    for match in LONG_NUMBER.finditer(text):
        if sum(c.isdigit() for c in match.group()) >= MIN_NUMBER_DIGITS:
            add("number", match.start(), match.end())
    return found


def _name_parts(text: str, accepted: list[Identifier]) -> list[Identifier]:
    """Find the parts of a full name used alone later, such as "Mr Perera"."""
    parts = {
        part
        for found in accepted
        if found.kind == "name" and " " in found.text
        for part in found.text.split()
    }
    extra = []
    for part in parts:
        for match in re.finditer(rf"\b{re.escape(part)}\b", text):
            extra.append(Identifier("name", match.start(), match.end(), part))
    return extra


def _resolve(candidates: list[Identifier]) -> list[Identifier]:
    rank = {kind: index for index, kind in enumerate(PRIORITY)}
    ordered = sorted(
        candidates, key=lambda f: (rank[f.kind], -(f.end - f.start), f.start)
    )
    accepted: list[Identifier] = []
    for candidate in ordered:
        if all(candidate.end <= f.start or candidate.start >= f.end for f in accepted):
            accepted.append(candidate)
    return accepted


def identify(text: str) -> list[Identifier]:
    """Return every finding in ``text``, without overlaps, in order."""
    accepted = _resolve(_candidates(text))
    accepted = _resolve(accepted + _name_parts(text, accepted))
    return sorted(accepted, key=lambda f: f.start)


def generalise_quasi(text: str, level: int) -> tuple[str, list[Identifier]]:
    """Generalise quasi-identifiers: level 1 occupations and conditions, level 2 ages.

    Returns the new text and what was generalised.
    """
    kinds = ("occupation", "condition") if level == 1 else ("age",)
    targets = [f for f in identify(text) if f.kind in kinds]
    out, last = [], 0
    for found in targets:
        out.append(text[last : found.start])
        out.append(_generalised(found))
        last = found.end
    out.append(text[last:])
    return "".join(out), targets


def _generalised(found: Identifier) -> str:
    if found.kind == "age":
        decade = int(found.text) // 10 * 10
        return f"{decade}-{decade + 9}"
    lowered = found.text.lower()
    table = OCCUPATIONS if found.kind == "occupation" else CONDITIONS
    for phrase in sorted(table, key=len, reverse=True):
        if lowered.startswith(phrase):
            return table[phrase]
    return "[generalised]"
