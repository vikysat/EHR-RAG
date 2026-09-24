"""Config loading and validation.

The config is a nested YAML mapping. Every key must be known (see ``SCHEMA``);
unknown or missing keys raise ``ConfigError``. Relative paths in the config are
resolved against the project root, defined as the parent of the directory that
holds the config file (``<root>/config/default.yaml`` -> ``<root>``).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

SECTIONS: tuple[str, ...] = ("History", "Findings", "Impression", "Recommendations")
BACKENDS: tuple[str, ...] = ("anthropic", "openai_compatible")
CONDITIONS: tuple[str, ...] = ("no_retrieval", "ehr_rag", "full_context")
EDIT_TYPES: tuple[str, ...] = ("lab_minor", "lab_major", "med_dose", "delete")

_NONE = type(None)
_INT = (int,)
_NUM = (int, float)
_STR = (str,)
_OPT_STR = (str, _NONE)
_BOOL = (bool,)
_LIST = (list,)

# Leaf values are tuples of allowed Python types. The special marker
# "queries" denotes the section -> list[str] mapping.
SCHEMA: dict[str, Any] = {
    "data": {"source": _STR, "root": _STR, "derived": _STR},
    "selection": {
        "min_distinct_labs": _INT,
        "min_abnormal_labs": _INT,
        "min_distinct_drugs": _INT,
        "min_diagnoses": _INT,
        "n_admissions": _INT,
    },
    "units": {"max_prior_admissions": _INT, "max_history_dx_per_admission": _INT},
    "retrieval": {
        "dense_query_model": _STR,
        "dense_doc_model": _STR,
        "rrf_k": _INT,
        "top_k_per_query": _INT,
        "max_units_per_section": _INT,
        "max_units_total": _INT,
        "rerank": _BOOL,
        "rerank_model": _STR,
        "relevance_export_pairs": _INT,
        "queries": "queries",
    },
    "generation": {
        "backend": _OPT_STR,
        "model": _OPT_STR,
        "base_url": _OPT_STR,
        "api_key_env": _OPT_STR,
        "temperature": _NUM,
        "max_tokens": _INT,
        "prompt_version": _STR,
        "conditions": _LIST,
        "max_context_tokens": _INT,
        "confirm_threshold": _INT,
        "endpoint_is_physionet_compliant": _BOOL,
    },
    "eval": {"nli_model": _STR, "manual_review_n": _INT, "seed": _INT},
    "perturbation": {
        "n_targets": _INT,
        "max_targets_per_admission": _INT,
        "edit_types": _LIST,
        "n_controls": _INT,
        "seed": _INT,
    },
}


class ConfigError(ValueError):
    """Raised when a config file is invalid."""


def _check_mapping(value: Any, schema: dict[str, Any], where: str) -> None:
    if not isinstance(value, dict):
        raise ConfigError(f"{where or 'config'}: expected a mapping, got {type(value).__name__}")
    unknown = sorted(set(value) - set(schema))
    if unknown:
        raise ConfigError(f"{where or 'config'}: unknown key(s) {unknown}")
    missing = sorted(set(schema) - set(value))
    if missing:
        raise ConfigError(f"{where or 'config'}: missing key(s) {missing}")
    for key, sub in schema.items():
        path = f"{where}.{key}" if where else key
        v = value[key]
        if isinstance(sub, dict):
            _check_mapping(v, sub, path)
        elif sub == "queries":
            _check_queries(v, path)
        else:
            # bool is a subclass of int; reject bools where numbers are expected.
            if isinstance(v, bool) and bool not in sub:
                raise ConfigError(f"{path}: expected {_type_names(sub)}, got bool")
            if not isinstance(v, sub):
                raise ConfigError(f"{path}: expected {_type_names(sub)}, got {type(v).__name__}")


def _type_names(types: tuple[type, ...]) -> str:
    return " or ".join("null" if t is _NONE else t.__name__ for t in types)


def _check_queries(value: Any, where: str) -> None:
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a mapping of section -> list of queries")
    unknown = sorted(set(value) - set(SECTIONS))
    if unknown:
        raise ConfigError(f"{where}: unknown section(s) {unknown}; allowed {list(SECTIONS)}")
    missing = [s for s in SECTIONS if s not in value]
    if missing:
        raise ConfigError(f"{where}: missing section(s) {missing}")
    for section, queries in value.items():
        if not isinstance(queries, list) or not queries or not all(isinstance(q, str) for q in queries):
            raise ConfigError(f"{where}.{section}: expected a non-empty list of strings")


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate structure, types and enumerated values. Raises ConfigError."""
    _check_mapping(cfg, SCHEMA, "")
    if cfg["data"]["source"] not in ("demo", "full"):
        raise ConfigError(f"data.source: must be 'demo' or 'full', got {cfg['data']['source']!r}")
    backend = cfg["generation"]["backend"]
    if backend is not None and backend not in BACKENDS:
        raise ConfigError(f"generation.backend: must be one of {list(BACKENDS)} or null, got {backend!r}")
    bad = [c for c in cfg["generation"]["conditions"] if c not in CONDITIONS]
    if bad:
        raise ConfigError(f"generation.conditions: unknown condition(s) {bad}")
    bad = [e for e in cfg["perturbation"]["edit_types"] if e not in EDIT_TYPES]
    if bad:
        raise ConfigError(f"perturbation.edit_types: unknown edit type(s) {bad}")
    for key in ("top_k_per_query", "max_units_per_section", "max_units_total", "rrf_k"):
        if cfg["retrieval"][key] < 1:
            raise ConfigError(f"retrieval.{key}: must be >= 1")


def validate_for_generation(cfg: dict[str, Any]) -> None:
    """Checks that must pass before any generation run. Raises ConfigError."""
    gen = cfg["generation"]
    if not gen["model"]:
        raise ConfigError("generation.model is not set; it has no default. Set it in the config.")
    if not gen["backend"]:
        raise ConfigError("generation.backend is not set; choose 'anthropic' or 'openai_compatible'.")
    if not gen["api_key_env"]:
        raise ConfigError("generation.api_key_env is not set; name the environment variable holding the API key.")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML config.

    The returned dict carries an extra private key ``_project_root`` (a string)
    used to resolve relative paths; it is not part of the schema.
    """
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ConfigError(f"{path}: empty config")
    validate_config(cfg)
    cfg = copy.deepcopy(cfg)
    cfg["_project_root"] = str(path.parent.parent)
    return cfg


def project_root(cfg: dict[str, Any]) -> Path:
    """Project root directory for this config."""
    return Path(cfg["_project_root"])


def resolve_path(cfg: dict[str, Any], p: str | Path) -> Path:
    """Resolve a (possibly relative) path against the project root."""
    p = Path(p)
    return p if p.is_absolute() else project_root(cfg) / p


def public_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Config without private keys, suitable for snapshotting."""
    return {k: copy.deepcopy(v) for k, v in cfg.items() if not k.startswith("_")}
