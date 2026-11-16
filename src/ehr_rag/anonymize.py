"""Anonymised admission labels for output that is meant to be shared.

With credentialed data (``data.source: full``) stdout, stderr and the logs made from them
never show a ``hadm_id`` or a ``subject_id``. An admission is shown as ``adm_001``,
``adm_002``...; a patient as ``subj_NNN``, with the number of the patient's first admission.
The real ids stay in the run folders: every run writes ``admission_aliases.csv``
(``alias,hadm_id,subject_id``). With Demo data nothing changes.

The numbers are stable. The frozen cohort lists come first (dev, then test, each in the
sorted order of its id file), so ``adm_007`` is the same admission in every command, shard
and run. Admissions outside the cohort follow in the order a command meets them.

Two layers:
  - per-admission lines and tables are written with ``label`` / ``admission``, which never
    return a real id while anonymising;
  - ``shared_output`` also replaces every known id in whatever reaches stdout or stderr
    (error messages, file names, tracebacks). It matches whole numbers of at least
    ``MIN_SCRUB_DIGITS`` digits; shorter ids cannot be told from counts (MIMIC-IV ids have 8).
"""

from __future__ import annotations

import csv
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ALIASES_FILE = "admission_aliases.csv"
MIN_SCRUB_DIGITS = 5
_NUMBER_RE = re.compile(rf"(?<!\d)\d{{{MIN_SCRUB_DIGITS},}}(?!\d)")


class Aliases:
    """Real id -> alias for one command. Disabled (the default), every method returns the real id."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._number: dict[int, int] = {}       # hadm_id -> n (adm_00n), in registration order
        self._subject: dict[int, int] = {}      # hadm_id -> subject_id, where known
        self._text: dict[str, str] = {}         # str(id) -> alias, for scrub
        self._run_dirs: list[Path] = []

    @classmethod
    def for_config(cls, cfg: dict[str, Any]) -> "Aliases":
        """Aliases for a command on this config: enabled only for ``data.source: full``.

        Numbering starts from the cohort id lists (hash-checked; skipped if absent); the
        stored unit headers add each admission's subject id and any admission outside the cohort.
        """
        a = cls(enabled=cfg["data"]["source"] == "full")
        if not a.enabled:
            return a
        from ehr_rag.config import ConfigError, resolve_path
        from ehr_rag.data.cohort import SPLITS, load_split_ids
        from ehr_rag.evidence.build_units import units_dir

        for split in SPLITS:
            try:
                ids = load_split_ids(cfg, split)
            except ConfigError:
                continue                    # no cohort here: admissions are numbered as they are met
            for h in ids:
                a.register(h)
        headers: list[tuple[int, int]] = []
        udir = units_dir(resolve_path(cfg, cfg["data"]["derived"]))
        for p in (udir.glob("*_header.json") if udir.is_dir() else []):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                headers.append((int(d["hadm_id"]), int(d["subject_id"])))
            except (OSError, ValueError, KeyError, TypeError):
                continue                    # an unreadable header names nobody
        for h, s in sorted(headers):
            a.register(h, s)
        return a

    def register(self, hadm_id: int, subject_id: int | None = None) -> None:
        """Give ``hadm_id`` the next number unless it has one; remember its patient if given."""
        if not self.enabled:
            return
        h = int(hadm_id)
        if h not in self._number:
            self._number[h] = len(self._number) + 1
            self._text[str(h)] = self._alias(h)
        if subject_id is not None and h not in self._subject:
            self._subject[h] = int(subject_id)
            self._text.setdefault(str(int(subject_id)), f"subj_{self._number[h]:03d}")

    def _alias(self, h: int) -> str:
        return f"adm_{self._number[h]:03d}"

    def label(self, hadm_id: int) -> Any:
        """``adm_00n`` (registering the admission if needed); the id itself when disabled."""
        if not self.enabled:
            return hadm_id
        self.register(hadm_id)
        return self._alias(int(hadm_id))

    def admission(self, hadm_id: int) -> str:
        """For progress lines: ``adm_00n``, or ``hadm <id>`` when disabled."""
        return str(self.label(hadm_id)) if self.enabled else f"hadm {hadm_id}"

    def scrub(self, text: str) -> str:
        """``text`` with every registered id (as a whole number, leading zeros allowed) replaced."""
        if not self.enabled or not self._text:
            return text
        return _NUMBER_RE.sub(lambda m: self._text.get(m.group(0).lstrip("0"), m.group(0)), text)

    def attach(self, run_dir: Path) -> None:
        """Keep the alias table in this run folder (written now and again by ``save``)."""
        if not self.enabled:
            return
        self._run_dirs.append(run_dir)
        self._write(run_dir)

    def save(self) -> None:
        """Rewrite the alias table in every attached run folder (ids registered since)."""
        for d in self._run_dirs:
            self._write(d)

    def _write(self, run_dir: Path) -> None:
        if not self._number or not run_dir.is_dir():
            return
        with (run_dir / ALIASES_FILE).open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["alias", "hadm_id", "subject_id"])
            w.writerows([self._alias(h), h, self._subject.get(h, "")] for h in self._number)


class _Scrubbed:
    """A text stream whose writes pass through ``Aliases.scrub``; everything else is the stream's own."""

    def __init__(self, stream: Any, aliases: Aliases) -> None:
        self._stream = stream
        self._aliases = aliases

    def write(self, s: Any) -> int:
        if not isinstance(s, str):
            return self._stream.write(s)
        self._stream.write(self._aliases.scrub(s))
        return len(s)

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


_active = Aliases()


def active() -> Aliases:
    """The aliases of the running command (disabled outside ``shared_output``)."""
    return _active


def label(hadm_id: int) -> Any:
    """``Aliases.label`` of the running command."""
    return _active.label(hadm_id)


def admission(hadm_id: int) -> str:
    """``Aliases.admission`` of the running command."""
    return _active.admission(hadm_id)


@contextmanager
def shared_output(aliases: Aliases) -> Iterator[Aliases]:
    """Make ``aliases`` the running command's; while enabled, scrub stdout and stderr.

    On exit the alias table is saved to the attached run folders and the streams are restored.
    """
    global _active
    previous = _active, sys.stdout, sys.stderr
    _active = aliases
    if aliases.enabled:
        sys.stdout, sys.stderr = _Scrubbed(sys.stdout, aliases), _Scrubbed(sys.stderr, aliases)
    try:
        yield aliases
    finally:
        try:
            aliases.save()
        finally:
            _active, sys.stdout, sys.stderr = previous
