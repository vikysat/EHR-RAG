"""Shared output on credentialed data: stdout and stderr never show a hadm_id or a subject_id.

Everything here is OBVIOUSLY FAKE: the tables of conftest.py with 8-digit ids (9xxxxxxx, a
range MIMIC-IV does not use), ``data.source: full`` and a fake two-admission cohort. Models
are the fakes of tests/fakes.py; nothing is downloaded.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

from conftest import FAKE_TABLES, REPO
from fakes import FakeBackend, FakeEncoder, FakeNLI, realistic_responder

SUBJECT = {1: 91000001, 2: 91000002}
ADMISSION = {11: 92000011, 12: 92000012, 21: 92000021}   # 92000011 is a prior admission, never a target
DEV, TEST = [ADMISSION[21]], [ADMISSION[12]]            # cohort order: adm_001 = 92000021, adm_002 = 92000012
ALL_IDS = [*SUBJECT.values(), *ADMISSION.values()]
ALIASES_CSV = "alias,hadm_id,subject_id\nadm_001,92000021,91000002\nadm_002,92000012,91000001\n"


def assert_no_ids(text: str) -> None:
    leaked = [i for i in ALL_IDS if str(i) in text]
    assert not leaked, f"ids {leaked} in shared output:\n{text[-3000:]}"


def _fake_full_tables() -> dict[str, list[list[Any]]]:
    out = {}
    for name, rows in FAKE_TABLES.items():
        maps = {rows[0].index(c): m for c, m in (("subject_id", SUBJECT), ("hadm_id", ADMISSION)) if c in rows[0]}
        out[name] = [rows[0]] + [[maps[i].get(v, v) if i in maps else v for i, v in enumerate(r)] for r in rows[1:]]
    return out


def make_project(root: Path) -> dict[str, Any]:
    """A FAKE full-data project: tables, config (``data.source: full``), prompts and a frozen cohort."""
    from ehr_rag.config import load_config
    from ehr_rag.data.cohort import write_split

    hosp = root / "fake_full" / "hosp"
    hosp.mkdir(parents=True)
    for name, rows in _fake_full_tables().items():
        with (hosp / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)
    raw = yaml.safe_load((REPO / "config" / "default.yaml").read_text(encoding="utf-8"))
    raw["data"] = {"source": "full", "root": str(root / "fake_full"), "derived": str(root / "derived_full")}
    raw["selection"].update(min_distinct_labs=1, min_abnormal_labs=0, min_distinct_drugs=1, min_diagnoses=1,
                            n_admissions=2)
    raw["cohort"]["out_dir"] = str(root / "cohort_out")
    raw["checker"].update(rules=str(REPO / "config" / "checker.yaml"),
                          lexicon=str(REPO / "config" / "checker_lexicon.yaml"))
    (root / "config").mkdir()
    cfg_path = root / "config" / "full_fake.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    tiny = copy.deepcopy(raw)
    tiny["generation"]["max_context_tokens"] = 1          # every full_context prompt is skipped
    tiny_path = root / "config" / "tiny_context.yaml"
    tiny_path.write_text(yaml.safe_dump(tiny), encoding="utf-8")
    (root / "prompts").mkdir()
    for p in (REPO / "prompts").glob("*.txt"):
        (root / "prompts" / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = load_config(cfg_path)
    write_split(cfg, "dev", DEV)
    write_split(cfg, "test", TEST)
    return {"root": root, "cfg": str(cfg_path), "tiny": str(tiny_path)}


def patch_models(mp: pytest.MonkeyPatch) -> None:
    import ehr_rag.cli as cli
    import ehr_rag.eval.nli as nli_mod
    import ehr_rag.generation.backends as backends
    from ehr_rag.retrieval.retrieve import Encoders

    class FakeEncoders(Encoders):
        def revisions(self) -> dict[str, str]:
            return {"fake-encoder": "fake-revision"}

    mp.setattr(backends, "make_backend", lambda gen: FakeBackend(realistic_responder))
    mp.setattr(nli_mod, "HFNLI", lambda name: FakeNLI())
    mp.setattr(cli, "_encoders", lambda cfg: FakeEncoders(query=FakeEncoder(), doc=FakeEncoder(), cross=None))


def latest(root: Path, command: str) -> str:
    runs = [p for p in (root / "runs").iterdir()
            if (p / "run_meta.json").exists()
            and json.loads((p / "run_meta.json").read_text(encoding="utf-8"))["command"] == command]
    return sorted(runs)[-1].name


def call(argv: list[str]) -> tuple[int, str, str]:
    """Run the CLI; return (exit status, stdout, stderr)."""
    from ehr_rag.cli import main

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


@pytest.fixture()
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    patch_models(monkeypatch)
    return make_project(tmp_path)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """Every command once on the fake full-data project; ``out[name]`` = (status, stdout, stderr)."""
    mp = pytest.MonkeyPatch()
    patch_models(mp)
    p = make_project(tmp_path_factory.mktemp("full_project"))
    root, cfg = p["root"], p["cfg"]
    out: dict[str, tuple[int, str, str]] = {}

    def run(name: str, command: str, *args: str, config: str = cfg) -> None:
        out[name] = call([command, "--config", config, *args])
        assert out[name][0] == 0, (name, out[name])

    split = ["--split", "dev", "test"]
    run("inspect-data", "inspect-data")
    run("select-admissions", "select-admissions")
    run("build-units", "build-units", *split)
    run("retrieve", "retrieve", *split)
    retrieve = latest(root, "retrieve")
    run("export-relevance", "export-relevance", *split)
    run("context-tokens", "context-tokens", *split)
    gen = "20990101-000000-generate"
    run("generate", "generate", *split, "--retrieval-run", retrieve, "--yes", "--run-id", gen, "--conditions",
        "no_retrieval", "ehr_rag", "full_context", "ehr_rag_enforced")
    run("generate-skip", "generate", *split, "--conditions", "full_context", config=p["tiny"])
    run("enforce-filter", "enforce-filter", "--generation-run", gen, "--run-id", "20990101-000001-enforce-filter")
    run("check-citations", "check-citations", "--generation-run", gen, "20990101-000001-enforce-filter",
        "--run-id", "20990101-000002-check-citations")
    run("evaluate", "evaluate", "--generation-run", gen, "--run-id", "20990101-000003-evaluate")
    refs = root / "fake_references.csv"
    refs.write_text("hadm_id,reference\n" + "".join(f"{h},FAKE discharge summary.\n" for h in DEV + TEST),
                    encoding="utf-8")
    run("text-metrics", "text-metrics", "--generation-run", gen, "--references", str(refs))
    run("audit-targets", "audit-targets", "--generation-run", gen, "--run-id", "20990101-000004-audit-targets")
    run("run-audit", "run-audit", "--targets-run", "20990101-000004-audit-targets", "--yes",
        "--run-id", "20990101-000005-run-audit")
    run("analyze-audit", "analyze-audit", "--audit-run", "20990101-000005-run-audit",
        "--run-id", "20990101-000006-analyze-audit")
    run("make-tables", "make-tables", "--evaluate-run", "20990101-000003-evaluate",
        "--check-run", "20990101-000002-check-citations", "--audit-run", "20990101-000006-analyze-audit")
    # labeling sheets, the labeler, retrieval metrics and the tables with the labels
    from itertools import cycle

    from ehr_rag.labeling import label_sheet

    run("labeling-sheets", "labeling-sheets", "--evaluate-run", "20990101-000003-evaluate", "--n", "2",
        "--n-relabel", "1", "--run-id", "20990101-000008-labeling-sheets")
    sheets = root / "runs" / "20990101-000008-labeling-sheets"
    keys = cycle("1342")
    label_sheet(sheets / "manual_review_sheet.csv", key=lambda: next(keys), out=lambda s: None)
    label_sheet(sheets / "relabel_sheet.csv", key=lambda: "3", out=lambda s: None)
    run("export-relevance-dev", "export-relevance", "--split", "dev", "--n-pairs", "1",
        "--run-id", "20990101-000009-export-relevance")
    rel = root / "runs" / "20990101-000009-export-relevance" / "relevance_to_label.csv"
    yn = cycle("yn")
    label_sheet(rel, key=lambda: next(yn), out=lambda s: None)
    run("retrieval-metrics", "retrieval-metrics", "--labels", str(rel), "--run-id", "20990101-000010-retrieval-metrics")
    run("make-tables-labels", "make-tables", "--label-sheet", str(sheets / "manual_review_sheet.csv"),
        "--label-key", str(sheets / "labeling_key.csv"), "--relabel-sheet", str(sheets / "relabel_sheet.csv"),
        "--retrieval-metrics-run", "20990101-000010-retrieval-metrics", "--run-id", "20990101-000011-make-tables")
    run("select-targets", "select-targets", "--generation-run", gen)
    run("run-pilot", "run-pilot", "--targets-run", latest(root, "select-targets"), "--yes",
        "--run-id", "20990101-000007-run-pilot")
    run("analyze-perturbation", "analyze-perturbation", "--pilot-run", "20990101-000007-run-pilot")
    yield {**p, "out": out, "generate": gen, "retrieve": retrieve}
    mp.undo()


# --------------------------------------------------------------------------- every command, full data


def test_no_command_prints_a_hadm_id_or_subject_id(pipeline: dict[str, Any]) -> None:
    for name, (_, out, err) in pipeline["out"].items():
        assert out.strip(), name
        assert_no_ids(out + err)


def test_build_units_counts_are_listed_per_alias(pipeline: dict[str, Any]) -> None:
    out = pipeline["out"]["build-units"][1]
    table = out.split("== Unit counts by type per admission")[1].split("\n\n")[0].splitlines()[1:]
    assert table[0].split()[0] == "admission" and "hadm_id" not in table[0]
    assert [row.split()[0] for row in table[1:]] == ["adm_001", "adm_002", "TOTAL"]
    assert int(table[1].split()[-1]) + int(table[2].split()[-1]) == int(table[3].split()[-1])   # counts kept


def test_retrieve_units_dropped_are_listed_per_alias(pipeline: dict[str, Any]) -> None:
    out = pipeline["out"]["retrieve"][1]
    assert "units dropped by max_units_total per admission: {'adm_001': 0, 'adm_002': 0}" in out


def test_generate_progress_lines_name_the_alias(pipeline: dict[str, Any]) -> None:
    out = pipeline["out"]["generate"][1]
    assert "  [1/8] no_retrieval adm_001: ok" in out and "  [2/8] no_retrieval adm_002: ok" in out
    assert "  [7/8] ehr_rag_enforced adm_001: ok; enforcement: kept" in out
    assert "hadm" not in out
    skip = pipeline["out"]["generate-skip"][1]
    assert "  SKIP full_context adm_001: " in skip and "  SKIP full_context adm_002: " in skip


def test_other_per_admission_lines_name_the_alias(pipeline: dict[str, Any]) -> None:
    out = {k: v[1] for k, v in pipeline["out"].items()}
    tokens = out["context-tokens"].splitlines()
    assert tokens[0].split()[0] == "admission" and [r.split()[0] for r in tokens[1:3]] == ["adm_001", "adm_002"]
    assert "ehr_rag_filtered adm_001: kept" in out["enforce-filter"]
    assert "ehr_rag adm_001__original: ok" in out["run-audit"]
    assert "ehr_rag_enforced adm_002__original: ok" in out["run-audit"]


def test_rows_and_queries_are_not_printed(pipeline: dict[str, Any]) -> None:
    out = {k: v[1] for k, v in pipeline["out"].items()}
    assert "first 5 rows" not in out["inspect-data"] and "== Row counts" in out["inspect-data"]
    assert "proposed: 2" in out["select-admissions"] and "richness" not in out["select-admissions"]
    assert "sampled " in out["export-relevance"] and "Fake" not in out["export-relevance"]   # queries name the dx
    for name in ("inspect-data", "select-admissions", "export-relevance"):
        assert "rows and text are not printed" in out[name], name


def test_real_ids_stay_in_the_run_folders(pipeline: dict[str, Any]) -> None:
    runs = pipeline["root"] / "runs"
    gen = runs / pipeline["generate"]
    assert sorted(p.name for p in (gen / "ehr_rag").iterdir()) == ["92000012", "92000021"]
    assert json.loads((gen / "run_meta.json").read_text(encoding="utf-8"))["hadm_ids"] == DEV + TEST
    build = runs / latest(pipeline["root"], "build-units")
    with (build / "unit_counts.csv").open(encoding="utf-8") as f:
        assert [r["hadm_id"] for r in csv.DictReader(f)] == ["92000021", "92000012"]
    for run in (build, gen, runs / pipeline["retrieve"], runs / "20990101-000005-run-audit"):
        assert (run / "admission_aliases.csv").read_text(encoding="utf-8") == ALIASES_CSV, run.name


def test_labeling_sheets_show_aliases_and_hide_the_condition(pipeline: dict[str, Any]) -> None:
    runs = pipeline["root"] / "runs"
    sheets = runs / "20990101-000008-labeling-sheets"
    for name in ("manual_review_sheet.csv", "relabel_sheet.csv", "labeling_key.csv"):
        assert_no_ids((sheets / name).read_text(encoding="utf-8"))
    with (sheets / "manual_review_sheet.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert sorted(r["admission"] for r in rows) == ["adm_001", "adm_002"]
    assert "condition" not in rows[0] and all(r["label"] for r in rows)
    tables = (runs / "20990101-000011-make-tables" / "tables.md").read_text(encoding="utf-8")
    assert "## Table 7." in tables and "## Table 8." in tables and "intra-annotator (4 labels)" in tables
    assert_no_ids(tables)
    out = pipeline["out"]["labeling-sheets"][1]
    assert "manual review sheet: 2 sentences from 2 admissions; re-label sheet: 1" in out


# --------------------------------------------------------------------------- numbering and error paths


def test_alias_numbers_follow_the_frozen_cohort_not_the_command(project: dict[str, Any]) -> None:
    cfg = project["cfg"]
    assert call(["build-units", "--config", cfg, "--split", "test"])[0] == 0
    rc, out, err = call(["generate", "--config", cfg, "--hadm-ids", str(TEST[0]), "--conditions", "no_retrieval"])
    assert rc == 0, err
    assert "  [1/1] no_retrieval adm_002: ok" in out            # second in dev + test, first in this command
    assert_no_ids(out + err)


def test_error_message_names_the_alias(project: dict[str, Any]) -> None:
    cfg, root = project["cfg"], project["root"]
    assert call(["build-units", "--config", cfg, "--split", "dev", "test"])[0] == 0
    assert call(["retrieve", "--config", cfg, "--split", "dev"])[0] == 0        # no result for the test admission
    rc, out, err = call(["generate", "--config", cfg, "--split", "dev", "test", "--conditions", "ehr_rag",
                         "--retrieval-run", latest(root, "retrieve"), "--yes"])
    assert rc == 4
    assert "generation failed: retrieval run has no result for hadm adm_002" in err and "adm_002.json" in err
    assert_no_ids(out + err)
    meta = json.loads((root / "runs" / latest(root, "generate") / "run_meta.json").read_text(encoding="utf-8"))
    assert str(TEST[0]) in meta["error"]                        # the run folder keeps the real id


def test_uncaught_error_traceback_is_scrubbed(project: dict[str, Any]) -> None:
    cfg, root = project["cfg"], project["root"]
    assert call(["build-units", "--config", cfg, "--split", "dev", "test"])[0] == 0
    (root / "derived_full" / "units" / f"{TEST[0]}.jsonl").unlink()
    rc, out, err = call(["retrieve", "--config", cfg, "--split", "test"])
    assert rc == 1                                              # as an uncaught exception would exit
    assert "Traceback" in err and "FileNotFoundError" in err and "adm_002.jsonl" in err
    assert_no_ids(out + err)


def test_guard_is_removed_after_the_command(project: dict[str, Any]) -> None:
    from ehr_rag import anonymize

    streams = sys.stdout, sys.stderr
    assert call(["build-units", "--config", project["cfg"], "--split", "dev"])[0] == 0
    assert (sys.stdout, sys.stderr) == streams and not anonymize.active().enabled


# --------------------------------------------------------------------------- Demo data: unchanged


def test_demo_output_still_shows_the_real_ids(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    patch_models(monkeypatch)
    cfg = str(env["root"] / "config" / "test.yaml")
    rc, out, _ = call(["generate", "--config", cfg, "--hadm-ids", "12", "21", "--conditions", "no_retrieval"])
    assert rc == 0 and "  [1/2] no_retrieval hadm 12: ok" in out and "adm_0" not in out
    assert not (env["root"] / "runs" / latest(env["root"], "generate") / "admission_aliases.csv").exists()
    rc, out, _ = call(["retrieve", "--config", cfg, "--hadm-ids", "12", "21", "--show", "0"])
    assert rc == 0 and "units dropped by max_units_total per admission: {12: 0, 21: 0}" in out


def test_demo_uncaught_errors_still_propagate(fake_config: Path) -> None:
    from ehr_rag.cli import main

    with pytest.raises(FileNotFoundError):
        main(["retrieve", "--config", str(fake_config), "--hadm-ids", "999"])


# --------------------------------------------------------------------------- the alias table itself


def test_aliases_are_numbered_in_registration_order() -> None:
    from ehr_rag.anonymize import Aliases

    a = Aliases(enabled=True)
    a.register(92000021, 91000002)
    a.register(92000012, 91000001)
    assert (a.label(92000021), a.label(92000012)) == ("adm_001", "adm_002")
    assert a.label(92000099) == "adm_003"                       # first met here: the next number, never the id
    assert a.admission(92000012) == "adm_002"


def test_disabled_aliases_return_the_id() -> None:
    from ehr_rag.anonymize import Aliases

    a = Aliases()
    a.register(92000021, 91000002)
    assert a.label(92000021) == 92000021 and a.admission(92000021) == "hadm 92000021"
    assert a.scrub("hadm 92000021") == "hadm 92000021"


def test_scrub_replaces_whole_known_ids_only() -> None:
    from ehr_rag.anonymize import Aliases

    a = Aliases(enabled=True)
    a.register(92000021, 91000002)
    text = "hadm 92000021, subject_id 91000002, units\\92000021.jsonl, 0092000021; 192000021 920000210 92000022"
    assert a.scrub(text) == "hadm adm_001, subject_id subj_001, units\\adm_001.jsonl, adm_001; 192000021 920000210 92000022"


def test_scrub_leaves_short_numbers_alone() -> None:
    from ehr_rag.anonymize import Aliases

    a = Aliases(enabled=True)
    a.register(12, 1)                                           # too short to tell from a count
    assert a.scrub("12 units of 1 patient") == "12 units of 1 patient"
    assert a.label(12) == "adm_001"                             # the explicit label never shows it


def test_shared_output_scrubs_both_streams_and_saves_the_table(tmp_path: Path,
                                                              capsys: pytest.CaptureFixture) -> None:
    from ehr_rag import anonymize

    a = anonymize.Aliases(enabled=True)
    a.register(92000021, 91000002)
    a.register(92000012, 91000001)
    with anonymize.shared_output(a):
        a.attach(tmp_path)
        print("out 92000021")
        print("err 91000001", file=sys.stderr)
        assert anonymize.label(92000012) == "adm_002" and anonymize.active() is a
    assert anonymize.label(92000012) == 92000012                 # nothing active any more
    cap = capsys.readouterr()
    assert (cap.out, cap.err) == ("out adm_001\n", "err subj_002\n")
    assert (tmp_path / "admission_aliases.csv").read_text(encoding="utf-8") == ALIASES_CSV


def test_aliases_for_config_start_from_the_cohort_lists(project: dict[str, Any]) -> None:
    from ehr_rag.anonymize import Aliases
    from ehr_rag.config import load_config

    a = Aliases.for_config(load_config(project["cfg"]))
    assert a.enabled and (a.label(DEV[0]), a.label(TEST[0])) == ("adm_001", "adm_002")
    raw = yaml.safe_load(Path(project["cfg"]).read_text(encoding="utf-8"))
    raw["data"]["source"] = "demo"
    Path(project["cfg"]).write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert not Aliases.for_config(load_config(project["cfg"])).enabled
