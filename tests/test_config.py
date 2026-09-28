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


def test_unset_generation_model_rejected(tmp_path: Path) -> None:
    cfg = _default()
    cfg["generation"]["model"] = None
    with pytest.raises(ConfigError, match="generation.model"):
        validate_for_generation(load_config(_write(tmp_path, cfg)))


def test_generation_ok_when_set(tmp_path: Path) -> None:
    cfg = _default()
    cfg["generation"].update(backend="anthropic", model="some-model", api_key_env="FAKE_KEY_ENV")
    validate_for_generation(load_config(_write(tmp_path, cfg)))


def test_default_config_is_hf_local_llama() -> None:
    cfg = load_config(DEFAULT)
    assert cfg["generation"]["backend"] == "hf_local"
    assert cfg["generation"]["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert cfg["generation"]["api_key_env"] is None
    validate_for_generation(cfg)  # no API key needed for hf_local


def test_api_backend_requires_api_key_env(tmp_path: Path) -> None:
    cfg = _default()
    cfg["generation"].update(backend="openai_compatible", model="some-model", api_key_env=None)
    with pytest.raises(ConfigError, match="api_key_env"):
        validate_for_generation(load_config(_write(tmp_path, cfg)))


def test_hf_local_requires_temperature_zero(tmp_path: Path) -> None:
    cfg = _default()
    cfg["generation"]["temperature"] = 0.7
    with pytest.raises(ConfigError, match="temperature"):
        validate_for_generation(load_config(_write(tmp_path, cfg)))


def test_unknown_backend_rejected(tmp_path: Path) -> None:
    cfg = _default()
    cfg["generation"]["backend"] = "fake"
    with pytest.raises(ConfigError, match="generation.backend"):
        load_config(_write(tmp_path, cfg))


@pytest.mark.parametrize("backend,base_url,refused", [
    ("anthropic", None, True),
    ("openai_compatible", "https://api.example.invalid/v1", True),
    ("openai_compatible", "http://localhost:8000/v1", False),
    ("hf_local", None, False),
])
def test_compliance_guard_full_data(tmp_path: Path, backend: str, base_url: str | None, refused: bool) -> None:
    cfg = _default()
    cfg["data"]["source"] = "full"
    cfg["generation"].update(backend=backend, base_url=base_url, api_key_env="FAKE_KEY_ENV")
    loaded = load_config(_write(tmp_path, cfg))
    if refused:
        with pytest.raises(ConfigError, match="PhysioNet"):
            validate_for_generation(loaded)
        loaded["generation"]["endpoint_is_physionet_compliant"] = True
    validate_for_generation(loaded)


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


def test_explicit_run_id_reuses_prepared_dir(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _default()))
    rid = "20260102-030405-generate-job42"
    pre = tmp_path / "runs" / rid
    pre.mkdir(parents=True)
    (pre / "nvidia-smi.txt").write_text("fake", encoding="utf-8")
    run = start_run(cfg, "generate", command="generate", run_id=rid)
    assert run.path == pre and (pre / "run_meta.json").exists() and (pre / "nvidia-smi.txt").exists()
    with pytest.raises(ValueError, match="already exists"):
        start_run(cfg, "generate", command="generate", run_id=rid)
    with pytest.raises(ValueError, match="run id"):
        start_run(cfg, "generate", command="generate", run_id="not-a-run-id")
