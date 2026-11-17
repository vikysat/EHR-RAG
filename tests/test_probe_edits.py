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


def _band(f: dict[str, Any], p: str) -> str:
    v, lo, hi = Decimal(f[p]["value"]), Decimal(f["ref_low"]), Decimal(f["ref_high"])
    return "high" if v > hi else "low" if v < lo else "normal" if lo < v < hi else "on a bound"


def test_within_flag_values_are_clamped_strictly_inside_their_band() -> None:
    # ref 3.3-5.1: +-10..20 % pushes 4.9 above 5.1 and 3.6 below 3.3 -> clamped to 5.0 / 3.4, never onto a bound
    near = lab("E1", "Potassium", [(T1, "3.6"), (T3, "4.9")])
    high = lab("E2", "Potassium", [(T1, "5.2"), (T3, "6.0")])          # 5.2 is one step above ref_high
    low = lab("E3", "Potassium", [(T1, "3.2"), (T3, "2.8")])           # 3.2 is one step below ref_low
    for u in (near, high, low):
        seen = set()
        for seed in range(30):
            r = within_flag_edit(u, [10, 20], f"s:{seed}")
            f0, f1 = u.fields, r.edited_fields
            for p in ("first", "max", "min", "last"):
                assert _band(f1, p) == _band(f0, p) != "on a bound"
                assert Decimal(f1[p]["value"]) != Decimal(f0[p]["value"]) and Decimal(f1[p]["value"]) > 0
            seen.add(r.edited_text)
        assert len(seen) > 1                                           # the seed still picks the factor
    up = within_flag_edit(near, [20, 20], "x")                         # only +-20 %: clamping does the work
    vals = {up.edited_fields[p]["value"] for p in ("first", "last")}
    assert vals in ({"4.3", "5.0"}, {"3.4", "3.9"})


def test_within_flag_skips_without_a_valid_in_band_value() -> None:
    # bounds cut the HIGH band away entirely: no value > 5.1 is allowed
    with pytest.raises(SkipEdit, match="no valid in-band value"):
        within_flag_edit(lab("E1", "Potassium", [(T1, "5.9")]), [10, 20], "s", (Decimal(0), Decimal("5.0")))
    # integer precision, ref 4.5-5.5: the band holds only the original 5, so no value can change
    only = lab("E1", "Potassium", [(T1, "5")], lo="4.5", hi="5.5")
    with pytest.raises(SkipEdit, match="inside its band"):
        within_flag_edit(only, [10, 20], "s")
