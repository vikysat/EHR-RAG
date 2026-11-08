"""Citation checker rules on FAKE units and sentences (one test group per rule)."""

from __future__ import annotations

import pytest

from checker_fakes import check, checker, fake_units, lab, T1, T3
from ehr_rag.checker.rules import CheckerError, Checker


def errs(line: str, **kw: object) -> list[str]:
    return check(line, **kw).error_types  # type: ignore[arg-type]


# --------------------------------------------------------------------------- statuses and coverage


def test_status_pass_fail_unchecked() -> None:
    a = check("Potassium peaked at 5.9. [E1]")
    assert a.status == "pass" and a.checkable and a.n_numbers == 1 and a.n_names == 1
    b = check("Potassium peaked at 6.9. [E1]")
    assert b.status == "fail" and b.error_types == ["number_mismatch"]
    c = check("The patient was monitored closely. [E1]")
    assert c.status == "unchecked" and not c.checkable and c.errors == []


# --------------------------------------------------------------------------- invalid_id, too_many_citations


def test_invalid_id_and_too_many_citations() -> None:
    assert errs("The patient was monitored. [E99]") == ["invalid_id"]
    lc = check("The patient was monitored. [E1, E2, E3, E5]")
    assert lc.error_types == ["too_many_citations"] and not lc.checkable
    assert errs("The patient was monitored. [E1, E2, E3]") == []


# --------------------------------------------------------------------------- number_mismatch


@pytest.mark.parametrize("line,ok", [
    ("Potassium peaked at 5.9. [E1]", True),
    ("Potassium peaked at 5.90 mmol/L. [E1]", True),        # formatting
    ("Potassium peaked at 6. [E1]", True),                  # rounding to the stated precision (5.9 -> 6)
    ("Potassium peaked at 5.95. [E1]", False),              # more precision than the unit: not a value
    ("Potassium peaked at 15.9. [E1]", False),
    ("Hemoglobin fell to 8.4. [E7]", True),                 # unit value 8.40
    ("Potassium peaked at 5.9. [E2]", False),               # the value is in an UNcited unit only
    ("Sodium was 130 on 2150-03-01 at 08:00. [E2]", True),  # ISO date and time ignored
    ("A 72-year-old was admitted for 5 days; sodium was 130. [E2]", True),   # age, duration ignored
    ("Sodium was 130 on 3/1/2150. [E2]", True),             # other date formats ignored
])
def test_number_rule(line: str, ok: bool) -> None:
    assert ("number_mismatch" not in errs(line)) is ok


def test_numbers_without_citation_fail() -> None:
    assert errs("Potassium peaked at 5.9.") == ["number_mismatch", "name_missing"]


# --------------------------------------------------------------------------- name_missing


@pytest.mark.parametrize("line,ok", [
    ("Furosemide was given. [E5]", True),
    ("Lasix was given. [E5]", True),                        # brand-name synonym
    ("Furosemide was given. [E1]", False),
    ("K was 4.3 and Na was 128. [E1, E2]", True),           # abbreviations
    ("Cr was 2.4. [E3]", True),
    ("Cr was 2.4. [E1]", False),
    ("WBC was 6.8. [E6]", True),                            # label "White Blood Cells"
    ("Hgb was 8.4. [E7]", True),
    ("The patient had AKI. [E8]", True),                    # term supported by the diagnosis title
    ("The patient had acute kidney failure. [E8]", True),
    ("The patient had AKI. [E5]", False),
    ("Hyperkalemia was present. [E1]", True),               # term supported by a cited potassium lab
])
def test_name_rule(line: str, ok: bool) -> None:
    assert ("name_missing" not in errs(line)) is ok


def test_abbreviations_are_case_sensitive() -> None:
    lc = check("The k value was fine. [E5]")        # lowercase 'k' is not potassium
    assert lc.n_names == 0 and lc.status == "unchecked"


# --------------------------------------------------------------------------- direction_mismatch


@pytest.mark.parametrize("line,ok", [
    ("Elevated potassium was seen. [E1]", True),            # max 5.9 [HIGH]
    ("Potassium was low. [E1]", False),
    ("Sodium was low at 128. [E2]", True),
    ("Sodium was normal. [E2]", False),
    ("Potassium was normal at 4.3. [E1]", True),            # the stated point decides
    ("Potassium was elevated at 4.3. [E1]", False),
    ("Hyperkalemia was present. [E1]", True),
    ("Hyponatremia was present. [E2]", True),
    ("Hypernatremia was present. [E2]", False),
    ("Anemia was noted. [E7]", True),
    ("Leukocytosis was noted. [E6]", False),
    ("Hyperglycemia was noted. [E10]", True),
    ("AKI with elevated creatinine. [E3, E8]", True),
    ("Elevated potassium was treated. [E4]", False),        # only the drug Potassium Chloride is cited
])
def test_direction_rule(line: str, ok: bool) -> None:
    assert ("direction_mismatch" not in errs(line)) is ok


def test_negation() -> None:
    assert "direction_mismatch" not in errs("There was no leukocytosis. [E6]")   # WBC normal
    assert "direction_mismatch" in errs("There was no hyperkalemia. [E1]")       # potassium peaked HIGH
    assert "direction_mismatch" in errs("Sodium was not low. [E2]")
    assert check("There was no leukocytosis. [E6]").status == "pass"


def test_direction_not_checkable_without_reference_range() -> None:
    lc = check("Troponin was elevated. [E9]")       # no reference range: direction not checkable
    assert lc.errors == [] and lc.items[0].n_directions == 0 and lc.n_names == 1


# --------------------------------------------------------------------------- trend_mismatch


@pytest.mark.parametrize("line,ok", [
    ("Creatinine was rising. [E3]", True),                  # 1.2 -> 2.4
    ("Creatinine was falling. [E3]", False),
    ("Creatinine was worsening. [E3]", True),               # creatinine gets worse upward
    ("Creatinine was improving. [E3]", False),
    ("Hemoglobin was trending down. [E7]", True),
    ("Sodium was trending. [E2]", True),                    # undirected: two measurements suffice
    ("Glucose was rising. [E10]", False),                   # one measurement only
    ("Creatinine was not improving. [E3]", True),           # negated
])
def test_trend_rule(line: str, ok: bool) -> None:
    assert ("trend_mismatch" not in errs(line)) is ok


# --------------------------------------------------------------------------- list sentences


def test_list_items_keep_opening_words() -> None:
    lc = check("Labs showed low Na 128 [E2], K 4.3 [E1], and Cr 2.4 [E3].")
    assert [it.text for it in lc.items] == ["Labs showed low Na 128", "Labs showed low K 4.3",
                                            "Labs showed low Cr 2.4"]
    assert [it.ids for it in lc.items] == [["E2"], ["E1"], ["E3"]]
    assert [bool(it.errors) for it in lc.items] == [False, True, True]     # 'low' carries to K and Cr
    ok = check("Labs showed Na 128 [E2], K 4.3 [E1], and Cr 2.4 [E3].")
    assert ok.status == "pass" and len(ok.items) == 3
    wrong = check("Labs showed Na 128 [E1], K 4.3 [E2].")                  # citations swapped
    assert wrong.error_types == ["number_mismatch", "name_missing"]


def test_too_many_citations_counted_per_item() -> None:
    assert errs("Na 128 [E2], K 4.3 [E1, E3, E4, E5].") == ["too_many_citations"]


# --------------------------------------------------------------------------- config validation


def test_lexicon_must_reference_known_lab_groups() -> None:
    import yaml

    from checker_fakes import REPO

    rules = yaml.safe_load((REPO / "config" / "checker.yaml").read_text(encoding="utf-8"))
    lex = yaml.safe_load((REPO / "config" / "checker_lexicon.yaml").read_text(encoding="utf-8"))
    Checker(rules, lex)
    lex["terms"]["bogus"] = {"labs": ["nope"], "direction": "high", "forms": ["bogusemia"]}
    with pytest.raises(CheckerError, match="unknown lab group"):
        Checker(rules, lex)
    rules["negation"]["cues"].append(False)              # an unquoted YAML 'no'
    with pytest.raises(CheckerError, match="strings only"):
        Checker(rules, yaml.safe_load((REPO / "config" / "checker_lexicon.yaml").read_text(encoding="utf-8")))


def test_checker_is_deterministic() -> None:
    units = fake_units() + [lab("E11", "Sodium, Urine", [(T1, "40"), (T3, "41")], lo=None, hi=None, fluid="Urine")]
    a = check("Labs showed low Na 128 [E2], K 4.3 [E1].", units=units).to_dict()
    b = check("Labs showed low Na 128 [E2], K 4.3 [E1].", units=units, ck=checker()).to_dict()
    assert a == b
