"""An append-only, hash-chained record of every request.

Entries never hold a protected value or a hash of one. Withheld items are kept
as their kind and, for numbers, the last 4 digits, as the project's deletion logs
do. Each entry's hash covers the entry and the previous hash, so removing or
changing an entry breaks the chain.

The cleared text that was sent is kept beside the chain, not in it, so erasing a
case can blank it without breaking the chain. By construction it holds no
identifiers, but if the gate ever missed one, erasing the case removes it here.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64
UNCHAINED = ("hash", "sent_text")


def _digest(entry: dict) -> str:
    chained = {key: value for key, value in entry.items() if key not in UNCHAINED}
    encoded = json.dumps(chained, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class AuditLog:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.entries: list[dict] = []

    def append(self, event: str, **fields) -> dict:
        entry = {
            "seq": len(self.entries),
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            **fields,
            "prev": self.entries[-1]["hash"] if self.entries else GENESIS,
        }
        entry["hash"] = _digest(entry)
        self.entries.append(entry)
        self._write()
        return entry

    def verify(self) -> bool:
        previous = GENESIS
        for index, entry in enumerate(self.entries):
            if entry.get("seq") != index or entry.get("prev") != previous:
                return False
            if entry.get("hash") != _digest(entry):
                return False
            previous = entry["hash"]
        return True

    def blank_case(self, case_ref: str) -> int:
        """Remove the sent text of a case's entries. The chain is unaffected."""
        count = 0
        for entry in self.entries:
            if entry.get("case_ref") == case_ref and entry.get("sent_text"):
                entry["sent_text"] = None
                count += 1
        self._write()
        return count

    def _write(self) -> None:
        if self.path is None:
            return
        # Rewritten whole, so blanking also leaves no copy in the file.
        lines = "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in self.entries)
        self.path.write_text(lines, encoding="utf-8")
