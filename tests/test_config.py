"""Config validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ehr_rag.config import ConfigError, load_config, validate_for_generation
from ehr_rag.runs import make_run_id, start_run

REPO = Path(__file__).resolve().parents[1]
DEFAULT = REPO / "config" / "default.yaml"


def _write(tmp_path: Path, cfg: dict) -> Path:
    d = tmp_path / "config"
    d.mkdir(exist_ok=True)
    p = d / "test.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _default() -> dict:
    return yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))


def test_default_config_loads() -> None:
    cfg = load_config(DEFAULT)
    assert cfg["retrieval"]["rrf_k"] == 60
    assert Path(cfg["_project_root"]) == REPO


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    cfg = _default()
    cfg["bogus"] = 1
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(_write(tmp_path, cfg))


def test_unknown_nested_key_rejected(tmp_path: Path) -> None:
    cfg = _default()
    cfg["retrieval"]["top_k"] = 5
    with pytest.raises(ConfigError, match=r"retrieval: unknown key"):
        load_config(_write(tmp_path, cfg))


def test_unknown_query_section_rejected(tmp_path: Path) -> None:
    cfg = _default()
    cfg["retrieval"]["queries"]["Plan"] = ["x"]
    with pytest.raises(ConfigError, match="unknown section"):
        load_config(_write(tmp_path, cfg))


def test_missing_key_rejected(tmp_path: Path) -> None:
    cfg = _default()
    del cfg["units"]["max_prior_admissions"]
    with pytest.raises(ConfigError, match="missing key"):
        load_config(_write(tmp_path, cfg))


def test_wrong_type_rejected(tmp_path: Path) -> None:
    cfg = _default()
    cfg["retrieval"]["rrf_k"] = "sixty"
    with pytest.raises(ConfigError, match="rrf_k"):
        load_config(_write(tmp_path, cfg))


def test_unset_generation_model_rejected() -> None:
    cfg = load_config(DEFAULT)
    assert cfg["generation"]["model"] is None
    with pytest.raises(ConfigError, match="generation.model"):
        validate_for_generation(cfg)


def test_generation_ok_when_set(tmp_path: Path) -> None:
    cfg = _default()
    cfg["generation"].update(backend="anthropic", model="some-model", api_key_env="FAKE_KEY_ENV")
    validate_for_generation(load_config(_write(tmp_path, cfg)))


def test_run_dir_has_config_snapshot(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _default()))
    run = start_run(cfg, "unittest", command="test", seed=13)
    assert run.path.parent == tmp_path / "runs"
    assert (run.path / "config.yaml").exists()
    snap = yaml.safe_load((run.path / "config.yaml").read_text(encoding="utf-8"))
    assert "_project_root" not in snap and snap["eval"]["seed"] == 13
    run2 = start_run(cfg, "unittest", command="test")
    assert run2.run_id != run.run_id


def test_run_id_format() -> None:
    from datetime import datetime

    assert make_run_id("abc", datetime(2026, 1, 2, 3, 4, 5)) == "20260102-030405-abc"
    with pytest.raises(ValueError):
        make_run_id("bad name")
