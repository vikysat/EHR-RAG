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
BACKENDS: tuple[str, ...] = ("anthropic", "openai_compatible", "hf_local")
API_BACKENDS: tuple[str, ...] = ("anthropic", "openai_compatible")
_LOCAL_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "::1")
CONDITIONS: tuple[str, ...] = ("no_retrieval", "ehr_rag", "full_context")
# Checker-enforced variants (no_retrieval has none: it cites nothing, so enforcement would drop
# every checkable sentence). *_enforced: one rewrite per failing sentence, then drop (generate);
# *_filtered: drop failing sentences, no model call (enforce-filter, CPU).
ENFORCED_CONDITIONS: tuple[str, ...] = ("ehr_rag_enforced", "full_context_enforced")
FILTERED_CONDITIONS: tuple[str, ...] = ("ehr_rag_filtered", "full_context_filtered")
GENERATE_CONDITIONS: tuple[str, ...] = CONDITIONS + ENFORCED_CONDITIONS
ALL_CONDITIONS: tuple[str, ...] = CONDITIONS + ENFORCED_CONDITIONS + FILTERED_CONDITIONS


def base_condition(condition: str) -> str:
    """``ehr_rag_enforced`` / ``ehr_rag_filtered`` -> ``ehr_rag``; base conditions map to themselves."""
    for suffix in ("_enforced", "_filtered"):
        if condition.endswith(suffix):
            return condition[: -len(suffix)]
    return condition
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
    "cohort": {
        "mimic_root": _STR,
        "note_root": _STR,
        "demo_root": _STR,
        "out_dir": _STR,
        "pool_fraction": _NUM,
        "seed": _INT,
        "min_los_days": _NUM,
        "max_los_days": _NUM,
        "min_prior_admissions": _INT,
        "min_lab_results": _INT,
        "n_dev": _INT,
        "n_test": _INT,
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
    "eval": {"nli_model": _STR, "manual_review_n": _INT, "seed": _INT, "bertscore_model": _STR,
             "bertscore_layer": _INT},
    "checker": {"rules": _STR, "lexicon": _STR, "rewrite_prompt": _STR},
    "audit": {
        "n_cited_per_admission": _INT,
        "n_null_per_admission": _INT,
        "seed": _INT,
        "outside_fraction": _NUM,
        "below_range_groups": _LIST,
        "bootstrap_resamples": _INT,
        "manual_check_n": _INT,
    },
    "probe": {
        "seed": _INT,
        "within_flag_percent": _LIST,
        "batch_size": _INT,
        "threshold_quantile": _NUM,
        "threshold_file": _STR,
        "bootstrap_resamples": _INT,
        "physiologic_bounds": (dict,),
        "related_drugs": (dict,),
    },
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
    bad = [c for c in cfg["generation"]["conditions"] if c not in GENERATE_CONDITIONS]
    if bad:
        raise ConfigError(f"generation.conditions: unknown condition(s) {bad}")
    bad = [e for e in cfg["perturbation"]["edit_types"] if e not in EDIT_TYPES]
    if bad:
        raise ConfigError(f"perturbation.edit_types: unknown edit type(s) {bad}")
    for key in ("top_k_per_query", "max_units_per_section", "max_units_total", "rrf_k"):
        if cfg["retrieval"][key] < 1:
            raise ConfigError(f"retrieval.{key}: must be >= 1")
    au = cfg["audit"]
    if au["outside_fraction"] <= 0:
        raise ConfigError("audit.outside_fraction: must be > 0")
    for key in ("n_cited_per_admission", "n_null_per_admission", "bootstrap_resamples", "manual_check_n"):
        if au[key] < 0:
            raise ConfigError(f"audit.{key}: must be >= 0")
    pr = cfg["probe"]
    wf = pr["within_flag_percent"]
    if len(wf) != 2 or not all(isinstance(x, int) and not isinstance(x, bool) for x in wf) or not 0 < wf[0] <= wf[1] < 100:
        raise ConfigError("probe.within_flag_percent: must be [low, high] whole percents with 0 < low <= high < 100")
    if pr["batch_size"] < 1 or pr["bootstrap_resamples"] < 0:
        raise ConfigError("probe.batch_size must be >= 1 and probe.bootstrap_resamples >= 0")
    if not 0 < pr["threshold_quantile"] < 1:
        raise ConfigError("probe.threshold_quantile: must be in (0, 1)")
    for g, e in pr["physiologic_bounds"].items():
        for b in (e.values() if isinstance(e, dict) else [e]):
            if not (isinstance(b, list) and len(b) == 2 and all(x is None or isinstance(x, (int, float)) for x in b)):
                raise ConfigError(f"probe.physiologic_bounds.{g}: expected [low, high] or a mapping uom -> [low, high]")
            if b[0] is not None and b[1] is not None and b[0] > b[1]:
                raise ConfigError(f"probe.physiologic_bounds.{g}: low > high")
    for g, e in pr["related_drugs"].items():
        if not isinstance(e, dict) or any(k not in ("high", "low") or not isinstance(v, list) for k, v in e.items()):
            raise ConfigError(f"probe.related_drugs.{g}: expected a mapping high/low -> list of drug names")
    co = cfg["cohort"]
    if not 0 < co["pool_fraction"] <= 1:
        raise ConfigError("cohort.pool_fraction: must be in (0, 1]")
    if co["min_los_days"] > co["max_los_days"]:
        raise ConfigError("cohort.min_los_days: must be <= cohort.max_los_days")
    for key in ("n_dev", "n_test", "min_prior_admissions", "min_lab_results"):
        if co[key] < 0:
            raise ConfigError(f"cohort.{key}: must be >= 0")


def validate_for_generation(cfg: dict[str, Any]) -> None:
    """Checks that must pass before any generation run. Raises ConfigError.

    The API-key check applies only to API backends; ``hf_local`` needs no key but
    requires temperature 0 (it always decodes greedily).
    """
    gen = cfg["generation"]
    if not gen["model"]:
        raise ConfigError("generation.model is not set; it has no default. Set it in the config.")
    if not gen["backend"]:
        raise ConfigError(f"generation.backend is not set; choose one of {list(BACKENDS)}.")
    if gen["backend"] in API_BACKENDS and not gen["api_key_env"]:
        raise ConfigError("generation.api_key_env is not set; name the environment variable holding the API key.")
    if gen["backend"] == "hf_local" and gen["temperature"] != 0:
        raise ConfigError("generation.temperature must be 0 for hf_local (greedy decoding only).")
    check_compliance(cfg)


def backend_is_external(gen: dict[str, Any]) -> bool:
    """True if the configured backend sends data off this machine.

    ``anthropic`` is always external; ``openai_compatible`` is external unless
    ``base_url`` points at localhost; ``hf_local`` runs in-process.
    """
    backend = gen["backend"]
    if backend == "hf_local":
        return False
    if backend == "openai_compatible" and gen["base_url"]:
        from urllib.parse import urlparse

        return urlparse(gen["base_url"]).hostname not in _LOCAL_HOSTS
    return True


COMPLIANCE_WARNING = (
    "WARNING: data.source is 'full' (credentialed MIMIC-IV). Credentialed data may only be sent "
    "to services permitted under the PhysioNet data use agreement."
)


def check_compliance(cfg: dict[str, Any]) -> None:
    """SPEC section 5: refuse full data + external endpoint unless declared compliant."""
    if cfg["data"]["source"] != "full" or not backend_is_external(cfg["generation"]):
        return
    if not cfg["generation"]["endpoint_is_physionet_compliant"]:
        raise ConfigError(
            COMPLIANCE_WARNING + " Refusing to run: generation.endpoint_is_physionet_compliant is false."
        )
    print(COMPLIANCE_WARNING)


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
