"""Find card numbers and NICs in chat text.

Stated values are matched strictly: a card must have the right shape and pass
Luhn. Model replies are matched loosely, because a garbled echo of a card fails
both tests and can still give most of the number away.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dragonscope.data import luhn_valid

CARD = re.compile(r"(?<!\d)\d{4}([ -]?)\d{4}\1\d{4}\1\d{4}(?!\d)")
NIC = re.compile(r"(?<![0-9A-Za-z])(?:\d{9}[VX]|\d{12})(?![0-9A-Za-z])")
# Fewer digits than this in a reply is ordinary text, such as "9 to 5".
MIN_REPLY_DIGITS = 4


@dataclass(frozen=True)
class Detection:
    kind: str  # "card", "nic" or "number"
    start: int
    end: int
    luhn_valid: bool | None = None  # Only meaningful for cards.


def digits_of(text: str) -> str:
    return "".join(c for c in text if c.isdigit())


def find_stated(text: str) -> list[Detection]:
    """Find well-formed values: Luhn-valid cards, and NICs of either format."""
    found = [
        Detection("card", match.start(), match.end(), True)
        for match in CARD.finditer(text)
        if luhn_valid(match.group())
    ]
    found += [Detection("nic", match.start(), match.end()) for match in NIC.finditer(text)]
    return sorted(found, key=lambda detection: detection.start)


def find_in_reply(text: str) -> Detection | None:
    """Return one span from the reply's first digit to its last, however garbled.

    Spanning the whole range also catches stray characters inside a number, such
    as ``4711 11A1 1111``, and leaves no digit fragments behind after Forget.
    """
    positions = [index for index, character in enumerate(text) if character.isdigit()]
    if len(positions) < MIN_REPLY_DIGITS:
        return None
    start, end = positions[0], positions[-1] + 1
    if text[end : end + 1] in ("V", "X"):
        end += 1
    return _classify_reply(text, start, end)


def _classify_reply(text: str, start: int, end: int) -> Detection:
    value = text[start:end]
    count = len(digits_of(value))
    if count == 16:
        return Detection("card", start, end, luhn_valid(value))
    if count == 12 or (count == 9 and value[-1] in "VX"):
        return Detection("nic", start, end)
    lowered = text.lower()
    if "card" in lowered:
        return Detection("card", start, end, False)
    if "nic" in lowered:
        return Detection("nic", start, end)
    return Detection("number", start, end)
