"""Probe edits: physiologic bounds, clamping in the audit edit, the within-flag edit (OBVIOUSLY FAKE units)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from checker_fakes import T1, T3, fake_units, lab
from ehr_rag.evidence.render import lab_marker
from ehr_rag.perturb.edit import SkipEdit, audit_edit, clamp, physiologic_bounds, within_flag_edit

REPO = Path(__file__).resolve().parents[1]
BELOW = ["hemoglobin", "hematocrit", "platelet", "albumin"]


def _cfg() -> dict[str, Any]:
    return yaml.safe_load((REPO / "config" / "default.yaml").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- edits and bounds


def _marks(f: dict[str, Any]) -> list[str]:
    return [lab_marker(f[p]["value"], f[p]["flag"], f["ref_low"], f["ref_high"]) for p in ("first", "max", "min", "last")]


def test_within_flag_edit_changes_every_value_and_keeps_markers() -> None:
    k = fake_units()[0]                                     # potassium 4.1 / 5.9 [HIGH] / 4.3
    r = within_flag_edit(k, [10, 20], "s:1")
    assert r.edit_type == "within_flag" and _marks(r.edited_fields) == _marks(k.fields)
    for p in ("first", "max", "min", "last"):
        old, new = Decimal(k.fields[p]["value"]), Decimal(r.edited_fields[p]["value"])
        assert new != old and Decimal("0.09") <= abs(new - old) / old <= Decimal("0.21")
        assert r.edited_fields[p]["flag"] == k.fields[p]["flag"]
    assert r.edited_fields["n_abnormal"] == k.fields["n_abnormal"]
    assert within_flag_edit(k, [10, 20], "s:1").edited_text == r.edited_text           # seeded
    tight = lab("E1", "Potassium", [(T1, "3.6"), (T3, "4.9")])       # +10% makes 4.9 HIGH, -10% makes 3.6 LOW
    with pytest.raises(SkipEdit):
        within_flag_edit(tight, [10, 20], "s")
    with pytest.raises(SkipEdit):                                    # every option leaves the bounds
        within_flag_edit(lab("E1", "Potassium", [(T1, "4.0")]), [10, 20], "s", (Decimal("3.9"), Decimal("4.1")))


def test_bounds_table_and_clamp() -> None:
    table = _cfg()["probe"]["physiologic_bounds"]
    assert physiologic_bounds(table, "potassium", "mEq/L") == (Decimal("1.0"), Decimal("10.0"))
    assert physiologic_bounds(table, "calcium", "mmol/L") == (Decimal("0.4"), Decimal("4"))
    assert physiologic_bounds(table, "calcium", "g/L") == (Decimal(0), None)        # uom without an entry
    assert physiologic_bounds(table, None, None) == (Decimal(0), None)              # never negative
    assert physiologic_bounds({"x": [-5, 3]}, "x", None) == (Decimal(0), Decimal(3))
    assert clamp(Decimal("12"), (Decimal(1), Decimal(10))) == 10 and clamp(Decimal("-1"), (Decimal(0), None)) == 0


def test_audit_edit_clamps_into_bounds_and_skips_when_the_flag_flip_is_lost() -> None:
    glu = lab("E10", "Glucose", [(T1, "90"), (T3, "95")], lo="70", hi="100")        # normal -> to_high 115
    assert audit_edit(glu, "glucose", 0.5, BELOW).edited_value == "115"
    assert audit_edit(glu, "glucose", 0.5, BELOW, (Decimal(0), Decimal(110))).edited_value == "110"
    with pytest.raises(SkipEdit, match="does not give marker"):
        audit_edit(glu, "glucose", 0.5, BELOW, (Decimal(0), Decimal(100)))          # clamped back into range
