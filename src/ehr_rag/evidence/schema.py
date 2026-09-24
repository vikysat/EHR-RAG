"""Evidence-unit schema, serialized as JSON lines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

UNIT_TYPES: tuple[str, ...] = ("diagnosis", "procedure", "lab", "medication", "history_diagnosis")
TIME_SCOPES: tuple[str, ...] = ("current", "prior")


@dataclass
class EvidenceUnit:
    """One citable piece of evidence for a target admission."""

    unit_id: str
    hadm_id: int
    subject_id: int
    unit_type: str
    time_scope: str
    timestamp: str | None
    fields: dict[str, Any]
    text: str
    source_table: str
    source_rows: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.unit_type not in UNIT_TYPES:
            raise ValueError(f"unknown unit_type {self.unit_type!r}")
        if self.time_scope not in TIME_SCOPES:
            raise ValueError(f"unknown time_scope {self.time_scope!r}")

    def to_json(self) -> str:
        """Serialize as one JSON line (stable key order)."""
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvidenceUnit":
        """Build from a parsed JSON object."""
        return cls(**d)


def write_units(path: Path, units: list[EvidenceUnit]) -> None:
    """Write units as JSON lines (UTF-8, ``\\n`` line endings)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for u in units:
            f.write(u.to_json() + "\n")


def read_units(path: Path) -> list[EvidenceUnit]:
    """Read units from a JSON-lines file."""
    with path.open("r", encoding="utf-8") as f:
        return [EvidenceUnit.from_dict(json.loads(line)) for line in f if line.strip()]
