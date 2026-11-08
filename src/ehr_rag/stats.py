"""Bootstrap confidence intervals that resample ADMISSIONS (the independent unit).

``bootstrap(groups, stat, n, seed)``: ``groups`` maps an admission id to that admission's data;
each resample draws len(groups) admissions with replacement (``random.Random(seed)``, admission
ids sorted first, so the result is deterministic), and ``stat`` maps the list of drawn
admissions' data to a number (or None, e.g. a zero denominator; such resamples are left out).
The interval is the 2.5th and 97.5th percentile (linear interpolation) of the resampled values.
"""

from __future__ import annotations

import random
from typing import Any, Callable

Stat = Callable[[list[Any]], "float | None"]


def percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile of an already sorted list (q in [0, 100])."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def bootstrap(groups: dict[Any, Any], stat: Stat, n: int, seed: int | str) -> dict[str, Any]:
    """Point estimate and 95% percentile CI. Returns {estimate, ci_low, ci_high, n_groups, n_resamples}."""
    keys = sorted(groups)
    data = [groups[k] for k in keys]
    est = stat(data) if data else None
    vals: list[float] = []
    if data and est is not None:
        rng = random.Random(seed)
        for _ in range(n):
            v = stat([data[rng.randrange(len(data))] for _ in data])
            if v is not None:
                vals.append(v)
    vals.sort()
    return {"estimate": est, "ci_low": percentile(vals, 2.5) if vals else None,
            "ci_high": percentile(vals, 97.5) if vals else None, "n_groups": len(keys), "n_resamples": len(vals)}


def ratio_stat(num: Callable[[Any], float], den: Callable[[Any], float]) -> Stat:
    """Pooled ratio sum(num) / sum(den) over the drawn admissions (None if the denominator is 0)."""
    def stat(items: list[Any]) -> float | None:
        d = sum(den(x) for x in items)
        return None if d == 0 else sum(num(x) for x in items) / d
    return stat


def fmt_ci(r: dict[str, Any], digits: int = 3) -> str:
    """'0.812 [0.701, 0.900]', or 'n/a'."""
    if r.get("estimate") is None:
        return "n/a"
    s = f"{r['estimate']:.{digits}f}"
    if r.get("ci_low") is not None:
        s += f" [{r['ci_low']:.{digits}f}, {r['ci_high']:.{digits}f}]"
    return s
