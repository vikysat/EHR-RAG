"""Verifier agreement between NLI labels and human labels (SPEC Phase 5).

As implemented: only rows with a non-blank ``human_support`` are used; values must
be one of supported / partial / unsupported / not_checkable (anything else stops).
``not_checkable`` rows are excluded from agreement and counted. Both sides are then
binarized: NLI ``supported`` iff nli_label is the entailment label; human
``supported`` iff human_support == "supported" (partial and unsupported count as not
supported). Raw agreement and Cohen's kappa are computed on the binary labels; the
full confusion matrix (NLI label x human label, including not_checkable) is reported
as well.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ehr_rag.eval.manual_review import HUMAN_SUPPORT_VALUES


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa for two equally long label lists (None if undefined)."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe == 1:
        return None
    return (po - pe) / (1 - pe)


def agreement(rows: list[dict[str, Any]], entailment_label: str = "entailment") -> dict[str, Any]:
    """Raw agreement, kappa and confusion matrix from labeled manual-review rows."""
    labeled = [r for r in rows if (r.get("human_support") or "").strip()]
    bad = sorted({r["human_support"].strip() for r in labeled} - set(HUMAN_SUPPORT_VALUES))
    if bad:
        raise ValueError(f"unknown human_support value(s) {bad}; allowed {list(HUMAN_SUPPORT_VALUES)}")
    confusion: dict[str, dict[str, int]] = {}
    for r in labeled:
        row = confusion.setdefault(r["nli_label"], {v: 0 for v in HUMAN_SUPPORT_VALUES})
        row[r["human_support"].strip()] += 1
    usable = [r for r in labeled if r["human_support"].strip() != "not_checkable"]
    nli_bin = ["supported" if r["nli_label"] == entailment_label else "not_supported" for r in usable]
    hum_bin = ["supported" if r["human_support"].strip() == "supported" else "not_supported" for r in usable]
    return {
        "n_rows": len(rows),
        "n_labeled": len(labeled),
        "n_not_checkable": len(labeled) - len(usable),
        "n_used": len(usable),
        "raw_agreement": (sum(x == y for x, y in zip(nli_bin, hum_bin)) / len(usable)) if usable else None,
        "cohen_kappa": cohen_kappa(nli_bin, hum_bin),
        "binary_confusion": {
            f"nli_{a}": {f"human_{b}": sum(1 for x, y in zip(nli_bin, hum_bin) if x == a and y == b)
                         for b in ("supported", "not_supported")}
            for a in ("supported", "not_supported")
        },
        "confusion_nli_x_human": confusion,
        "binarization": "NLI supported = entailment; human supported = 'supported' (partial/unsupported = "
                        "not supported); not_checkable excluded",
    }
