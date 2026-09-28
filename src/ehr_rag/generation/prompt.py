"""Prompt construction from versioned templates in ``prompts/``.

Files for prompt version ``<v>`` (never edited in place once released):
  ``<v>.txt``                 ehr_rag and full_context; placeholders {header}, {evidence}
  ``<v>_no_retrieval.txt``    no_retrieval; placeholder {header} only
  ``<v>_format_reminder.txt`` appended once on a format retry; placeholder {problems}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ehr_rag.evidence.schema import EvidenceUnit

_HEADER_FIELDS: tuple[tuple[str, str], ...] = (
    ("age", "Age"),
    ("gender", "Gender"),
    ("admission_type", "Admission type"),
    ("admission_location", "Admission location"),
    ("admittime", "Admitted"),
    ("dischtime", "Discharged"),
    ("los_days", "Length of stay (days)"),
)


@dataclass(frozen=True)
class PromptTemplates:
    """The three template texts of one prompt version."""

    version: str
    evidence: str
    no_retrieval: str
    format_reminder: str


def load_templates(prompts_dir: Path, version: str) -> PromptTemplates:
    """Read and sanity-check the templates for ``version``."""
    def read(suffix: str, required: tuple[str, ...], forbidden: tuple[str, ...] = ()) -> str:
        p = prompts_dir / f"{version}{suffix}.txt"
        if not p.exists():
            raise FileNotFoundError(f"prompt template not found: {p}")
        text = p.read_text(encoding="utf-8")
        for ph in required:
            if ph not in text:
                raise ValueError(f"{p.name}: missing placeholder {ph}")
        for ph in forbidden:
            if ph in text:
                raise ValueError(f"{p.name}: must not contain placeholder {ph}")
        return text

    return PromptTemplates(
        version=version,
        evidence=read("", ("{header}", "{evidence}")),
        no_retrieval=read("_no_retrieval", ("{header}",), ("{evidence}",)),
        format_reminder=read("_format_reminder", ("{problems}",)),
    )


def render_header(header: dict[str, Any]) -> str:
    """Plain labeled lines for the non-citable patient header (identifiers omitted)."""
    def fmt(key: str, v: Any) -> str:
        if v is None:
            return "unknown"
        if key in ("admittime", "dischtime"):
            return str(v).replace("T", " ")[:16]
        return str(v)

    return "\n".join(f"{label}: {fmt(key, header.get(key))}" for key, label in _HEADER_FIELDS)


def evidence_line(unit_id: str, unit_type: str, text: str) -> str:
    """One evidence-block line: ``[E12] (lab) <text>``."""
    return f"[{unit_id}] ({unit_type}) {text}"


def evidence_block(units: list[EvidenceUnit]) -> str:
    """Evidence block in the given (retrieval) order."""
    return "\n".join(evidence_line(u.unit_id, u.unit_type, u.text) for u in units)


def build_prompt(templates: PromptTemplates, header: dict[str, Any], units: list[EvidenceUnit] | None) -> str:
    """Fill a template. ``units=None`` selects the no_retrieval template (header only)."""
    h = render_header(header)
    if units is None:
        return templates.no_retrieval.replace("{header}", h)
    return templates.evidence.replace("{header}", h).replace("{evidence}", evidence_block(units))


def with_format_reminder(templates: PromptTemplates, prompt: str, problems: list[str]) -> str:
    """The original prompt followed by the format reminder listing ``problems``."""
    listed = "\n".join(f"- {p}" for p in problems)
    return prompt.rstrip("\n") + "\n\n" + templates.format_reminder.replace("{problems}", listed)
