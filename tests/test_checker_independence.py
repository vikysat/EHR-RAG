"""Evaluation stays independent of the citation checker; checker files are hashed into every run."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def _module_file(name: str) -> Path | None:
    p = SRC / Path(*name.split("."))
    for cand in (p.with_suffix(".py"), p / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _imports(path: Path) -> set[str]:
    """Every ehr_rag module imported anywhere in the file (including lazy imports inside functions)."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names if a.name.startswith("ehr_rag")}
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("ehr_rag"):
            out.add(node.module)
            out |= {f"{node.module}.{a.name}" for a in node.names if _module_file(f"{node.module}.{a.name}")}
    return out


def _closure(roots: list[str]) -> set[str]:
    seen: set[str] = set()
    todo = list(roots)
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        f = _module_file(m)
        if f is not None:
            todo.extend(_imports(f))
    return seen


def test_evaluation_never_imports_the_checker() -> None:
    roots = [f"ehr_rag.eval.{p.stem}" for p in (SRC / "ehr_rag" / "eval").glob("*.py") if p.stem != "__init__"]
    reach = _closure(roots + ["ehr_rag.runs"])
    assert "ehr_rag.eval.evaluate" in reach and "ehr_rag.eval.nli" in reach
    assert not [m for m in reach if m.startswith("ehr_rag.checker")]


def test_evaluate_command_uses_no_checker() -> None:
    """The evaluate subcommand's own lazy imports do not reach the checker either."""
    import inspect

    from ehr_rag.cli import cmd_evaluate

    tree = ast.parse(inspect.getsource(cmd_evaluate))
    mods = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
    assert mods and not [m for m in _closure(mods) if m.startswith("ehr_rag.checker")]


def test_checker_hashes_in_every_run_and_frozen_guard(env: dict[str, Any], capsys: pytest.CaptureFixture) -> None:
    from ehr_rag.cli import main
    from ehr_rag.runs import checker_hashes

    cfg = env["cfg"]
    h = checker_hashes(cfg)
    assert h and all(h[f"{k}_sha256"] for k in ("rules", "lexicon", "rewrite_prompt"))
    runs = Path(cfg["_project_root"]) / "runs"
    for d in runs.iterdir():                                   # every run started by the fixture
        meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["checker"]["combined_sha256"] == h["combined_sha256"]
    cfg_path = str(Path(cfg["_project_root"]) / "config" / "test.yaml")
    base = ["generate", "--config", cfg_path, "--retrieval-run", env["retrieval_run"], "--hadm-ids", "12"]
    assert main(base + ["--checker-sha256", "0" * 64]) == 2
    assert "checker files changed" in capsys.readouterr().err
    assert main(["validate-config", "--config", cfg_path]) == 0
    assert h["combined_sha256"] in capsys.readouterr().out


def test_hash_changes_when_the_lexicon_changes(env: dict[str, Any], tmp_path: Path) -> None:
    import copy

    from ehr_rag.runs import checker_hashes

    cfg = copy.deepcopy(env["cfg"])
    lex = tmp_path / "lexicon.yaml"
    lex.write_text(Path(cfg["checker"]["lexicon"]).read_text(encoding="utf-8") + "\n# tuned\n", encoding="utf-8")
    before = checker_hashes(cfg)["combined_sha256"]
    cfg["checker"]["lexicon"] = str(lex)
    assert checker_hashes(cfg)["combined_sha256"] != before
