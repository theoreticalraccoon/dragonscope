"""Placeholders for one request, and the re-linker that undoes them locally.

HIPAA's rule for re-identification codes asks that a code is not derived from
the person's data, is used for nothing else, and that the way back is never
disclosed. So codes here are random letters from ``secrets``, a map lives for
one request, and it never leaves this process.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field

from dragonscope.gateway.identifiers import DIRECT, GENERALISED, Identifier, identify

PREFIX = {
    "name": "PERSON",
    "card": "CARD",
    "nic": "NIC",
    "phone": "PHONE",
    "email": "EMAIL",
    "record": "RECORD",
    "number": "NUMBER",
    "place": "PLACE",
}
# No I or O, so a code is not misread as a digit.
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ"
CODE_LENGTH = 4
CODE = re.compile(r"\b(?:" + "|".join(PREFIX.values()) + r")_[A-Z]{4}\b")


class PlaceholderMap:
    """Maps random codes to real values for one request. Call ``destroy`` after."""

    def __init__(self) -> None:
        self._by_value: dict[tuple[str, str], str] = {}
        self._by_code: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._by_code)

    def code_for(self, kind: str, value: str) -> str:
        key = (kind, value)
        if key not in self._by_value:
            while True:
                code = f"{PREFIX[kind]}_" + "".join(
                    secrets.choice(ALPHABET) for _ in range(CODE_LENGTH)
                )
                if code not in self._by_code:
                    break
            self._by_value[key] = code
            self._by_code[code] = value
        return self._by_value[key]

    def codes(self) -> list[str]:
        return list(self._by_code)

    def relink(self, text: str) -> "Relinked":
        """Swap codes back to real values, flagging what the map does not explain."""
        invented = sorted(
            {f.kind for f in identify(text, guess_names=False) if f.kind in DIRECT}
        )
        unknown = sorted({code for code in CODE.findall(text) if code not in self._by_code})
        restored = CODE.sub(lambda m: self._by_code.get(m.group(), m.group()), text)
        return Relinked(restored, invented, len(unknown))

    def destroy(self) -> None:
        self._by_value.clear()
        self._by_code.clear()


@dataclass
class Relinked:
    text: str
    invented: list[str] = field(default_factory=list)  # Kinds of identifier the model wrote.
    unknown_codes: int = 0  # Codes in the answer that this request never issued.


def deidentify(text: str, placeholders: PlaceholderMap) -> tuple[str, list[Identifier]]:
    """Replace direct identifiers and generalise dates and old ages.

    Quasi-identifiers are returned but left in place; the gate decides on them.
    """
    found = identify(text)
    full_names = [f.text for f in found if f.kind == "name" and " " in f.text]
    out, last = [], 0
    for item in found:
        out.append(text[last : item.start])
        if item.kind == "name" and " " not in item.text:
            # "Mr Perera" gets the same code as "Nimal Perera" earlier.
            whole = next((n for n in full_names if item.text in n.split()), item.text)
            out.append(placeholders.code_for("name", whole))
        elif item.kind in DIRECT:
            out.append(placeholders.code_for(item.kind, item.text))
        elif item.kind in GENERALISED:
            out.append(item.replacement or "")
        else:
            out.append(item.text)
        last = item.end
    out.append(text[last:])
    return "".join(out), found
