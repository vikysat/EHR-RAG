"""Run directories: ``runs/<run_id>/`` with ``run_id = YYYYMMDD-HHMMSS-<name>``.

Every run writes ``config.yaml`` (the resolved config) and ``run_meta.json``
(command, argv, seed, timestamps, platform and package versions). ``finish``
adds the end timestamp and status.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from ehr_rag.config import project_root, public_config

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_TRACKED_PACKAGES = (
    "numpy", "pandas", "pyyaml", "rank_bm25", "faiss-cpu", "torch",
    "transformers", "sentence-transformers",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _git_commit(root: Path) -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return res.stdout.strip() or None


def make_run_id(name: str, now: datetime | None = None) -> str:
    """Build ``YYYYMMDD-HHMMSS-<name>``."""
    if not _NAME_RE.match(name):
        raise ValueError(f"run name must match {_NAME_RE.pattern}, got {name!r}")
    now = now or datetime.now()
    return f"{now:%Y%m%d-%H%M%S}-{name}"


@dataclass
class Run:
    """An open run directory."""

    run_id: str
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)

    def write_json(self, relpath: str, obj: Any) -> Path:
        """Write ``obj`` as pretty JSON under the run directory."""
        p = self.path / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return p

    def finish(self, status: str = "ok", **extra: Any) -> None:
        """Record end time, status and any extra fields in run_meta.json."""
        self.meta["finished_at"] = _now_iso()
        self.meta["status"] = status
        self.meta.update(extra)
        self.write_json("run_meta.json", self.meta)


def start_run(cfg: dict[str, Any], name: str, command: str, seed: int | None = None) -> Run:
    """Create ``runs/<run_id>/`` and write the config snapshot and metadata.

    If a directory with the same id already exists (two runs of the same name
    within one second), a numeric suffix ``-2``, ``-3``... is appended.
    """
    runs_dir = project_root(cfg) / "runs"
    base = make_run_id(name)
    run_id, i = base, 1
    while (runs_dir / run_id).exists():
        i += 1
        run_id = f"{base}-{i}"
    path = runs_dir / run_id
    path.mkdir(parents=True)
    with (path / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(public_config(cfg), f, sort_keys=False, allow_unicode=True)
    meta: dict[str, Any] = {
        "run_id": run_id,
        "command": command,
        "argv": sys.argv,
        "seed": seed,
        "started_at": _now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(project_root(cfg)),
        "packages": _package_versions(),
    }
    run = Run(run_id=run_id, path=path, meta=meta)
    run.write_json("run_meta.json", meta)
    return run
