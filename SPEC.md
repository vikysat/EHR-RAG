# SPEC.md — EHR-Aware Clinical RAG with Citation-Faithfulness Probes

This document is the single source of truth for this codebase. Read all of it before writing any code.

---

## 0. Operating rules (non-negotiable)

1. **Work phase by phase, in order.** At every `GATE`, stop, report exactly what the gate asks for, and wait for my explicit reply `continue` before starting the next phase.
2. **Never invent schema.** Before using any table, read its actual header and 5 rows. If a file, column, or value format differs from this spec, STOP and report. Do not guess, rename, or work around silently.
3. **No silent fallbacks.** If a model fails to download, an API call fails, or a library is missing, stop and report. Never substitute a different model, dataset, library, or method without asking.
4. **No fabricated data in the real pipeline.** No mock patients, no placeholder reference reports, no synthetic notes. Test fixtures in `tests/` are allowed and must be obviously fake.
5. **Do not build anything in §7 (Out of scope).** If you believe something outside this spec is needed, ask first.
6. **Everything reproducible.** Every run saves: config snapshot, seed, model id as returned by the API, timestamps, raw prompts, raw outputs.
7. **Never commit** data, API keys, model weights, embedding caches, or run outputs.
8. **Keep it simple.** Plain Python modules plus CLI entrypoints. No web UI, no database, no LangChain/LlamaIndex or other orchestration frameworks. Type hints on all functions; docstrings on public ones.
9. **Tests.** Write the tests listed in §6. Run `pytest` before every gate and report the result.
10. **Commit at the end of each phase** with message `Phase N: <summary>`.

---

## 1. Project summary

- **Input:** one hospital admission from the MIMIC-IV Clinical Database Demo (structured data only).
- **Output:** a structured clinical report with four sections (History, Findings, Impression, Recommendations) where every sentence cites evidence-unit IDs such as `[E12]`.
- **Research question:** are the citations *faithful*, i.e. do they track the evidence the model actually used? Tested by counterfactual perturbation: edit one cited fact, regenerate with retrieval frozen, and check whether the claim and its citation follow the edit.

---

## 2. Environment and data

- Python 3.10+. Dependencies: pandas, numpy, pyyaml, rank_bm25, faiss-cpu (faiss-gpu if a GPU is present), torch, transformers, sentence-transformers, pytest, tqdm, plus the SDK for the configured generation backend. Pin versions in `requirements.txt`.
- Data root comes from config `data.root`. **Inspect the directory tree first.** MIMIC-IV versions differ in module layout (e.g. whether patients/admissions sit under `hosp/` or `core/`). Report the tree at Gate 1.
- MIMIC dates are deliberately shifted into the future (2100s and later). This is not an error. Do not correct it.
- `data.source` is `demo` or `full`. See §5 (compliance guard).

---

## 3. Repository layout

```
ehr-rag/
  SPEC.md
  CLAUDE.md
  README.md
  requirements.txt
  config/default.yaml
  config/selected_admissions.yaml      # written in Phase 1, edited by me
  prompts/report_v1.txt
  src/ehr_rag/
    cli.py
    config.py
    data/load.py
    data/select_admissions.py
    evidence/schema.py
    evidence/render.py
    evidence/build_units.py
    retrieval/bm25.py
    retrieval/dense.py
    retrieval/fuse.py
    retrieval/retrieve.py
    retrieval/relevance.py
    generation/prompt.py
    generation/backends.py
    generation/generate.py
    eval/parse.py
    eval/nli.py
    eval/citations.py
    eval/text_metrics.py
    eval/manual_review.py
    eval/agreement.py
    perturb/select_targets.py
    perturb/edit.py
    perturb/run_pilot.py
    perturb/analyze.py
  tests/
  notebooks/demo.ipynb
  data/            # gitignored
  cache/           # gitignored
  runs/            # gitignored
```

---

## 4. Phases

### Phase 0 — Scaffold

- Create the layout above, `requirements.txt`, `config/default.yaml` (see §8), `.gitignore` (data/, cache/, runs/, .env, *.faiss, model caches), and a README stub.
- Config loader with validation: unknown keys raise an error.
- Run convention: `runs/<run_id>/` where `run_id = YYYYMMDD-HHMMSS-<name>`. Each run writes a copy of the resolved config.
- **Acceptance:** `pytest` passes; `python -m ehr_rag.cli --help` lists subcommands.

### Phase 1 — Data inspection and admission selection

Load: patients, admissions, labevents, d_labitems, prescriptions, diagnoses_icd, d_icd_diagnoses, procedures_icd, d_icd_procedures.

Expected columns (verify against real headers; mismatch → STOP):
- patients: subject_id, gender, anchor_age, anchor_year
- admissions: subject_id, hadm_id, admittime, dischtime, admission_type, admission_location, discharge_location
- labevents: labevent_id, subject_id, hadm_id, itemid, charttime, value, valuenum, valueuom, ref_range_lower, ref_range_upper, flag
- d_labitems: itemid, label, fluid, category
- prescriptions: subject_id, hadm_id, starttime, stoptime, drug, dose_val_rx, dose_unit_rx, route
- diagnoses_icd: subject_id, hadm_id, seq_num, icd_code, icd_version
- d_icd_diagnoses / d_icd_procedures: icd_code, icd_version, long_title
- procedures_icd: subject_id, hadm_id, seq_num, chartdate, icd_code, icd_version

Lab linkage: `labevents.hadm_id` can be null. Count rows linked by hadm_id; separately count null-hadm_id rows whose charttime falls within [admittime, dischtime] of an admission for the same subject. Include the latter, marked `linked_by = "time_window"`. Report both counts.

For every admission compute: distinct lab tests, abnormal lab results (outside ref range, or `flag == "abnormal"` when ref range is missing), distinct drugs, diagnoses, procedures, length of stay, and number of **prior** admissions for the same patient (dischtime before this admittime).

Eligibility (config-driven): ≥10 distinct lab tests, ≥3 abnormal results, ≥5 distinct drugs, ≥3 diagnoses. Ranking: prefer admissions with ≥1 prior admission, then by richness. At most one target admission per patient (the latest eligible one).

Outputs: `data/derived/admission_candidates.csv` (all admissions, metrics, eligible flag) and `config/selected_admissions.yaml` (top `selection.n_admissions` proposed).

**GATE 1 — report:** data directory tree; any column mismatches; row counts per table; lab linkage counts; number of eligible admissions; the proposed list with its metrics. Wait for my approval or edits to the list.

### Phase 2 — Evidence units

Schema (`evidence/schema.py`, dataclass, serialized as JSON lines):

```
unit_id: str          # "E1", "E2", ... unique within one target admission
hadm_id: int          # the TARGET admission this unit belongs to
subject_id: int
unit_type: str        # lab | medication | diagnosis | procedure | history_diagnosis
time_scope: str       # current | prior
timestamp: str|None   # ISO 8601; for summary units, the earliest relevant time
fields: dict          # structured values the text is rendered from
text: str             # rendered text shown to the model
source_table: str
source_rows: list     # primary keys of underlying rows (e.g. labevent_id list)
```

**Rendering rule (critical for Phase 6):** `text` is produced ONLY by a pure function `render(unit_type, fields) -> str` in `evidence/render.py`. Nothing ever edits `text` directly. Perturbations edit `fields` and re-render.

Unit construction:
- **lab**: one unit per (itemid, target admission). Fields: label, fluid, uom, ref_low, ref_high, n_measurements, first/last/min/max (each with value and time), n_abnormal. Render example:
  `Potassium (Blood), mmol/L, ref 3.3–5.1: 7 measurements. First 4.1 (2150-03-01 08:00); max 5.9 [HIGH] (2150-03-02 14:30); min 3.9 (2150-03-03 06:00); last 4.3 (2150-03-04 06:00).`
  `[HIGH]`/`[LOW]` are computed from the ref range; if the ref range is missing, use the `flag` column; if neither exists, no marker. Keep the original decimal precision of values. Text-only results (valuenum null): keep a unit showing the last text value; mark `fields.numeric = false` (excluded from numeric perturbation).
- **medication**: one unit per unique (drug, dose_val_rx, dose_unit_rx, route) within the admission. Fields include earliest start, latest stop, n_orders. Render example:
  `Furosemide 40 mg IV, ordered 3 times, 2150-03-01 to 2150-03-03.`
- **diagnosis**: one unit per ICD code of the target admission. Render example:
  `Diagnosis (priority 1 of 9): Acute kidney failure, unspecified (ICD-10 N17.9).`
- **procedure**: analogous to diagnosis, with chartdate.
- **history_diagnosis**: diagnoses from the same patient's admissions with dischtime before the target admittime. Cap at `units.max_history_dx_per_admission` per prior admission (by priority) and `units.max_prior_admissions` most recent prior admissions. Render example:
  `Prior admission (2149-11-02): Type 2 diabetes mellitus without complications (ICD-10 E11.9).`

Header (NOT an evidence unit, never citable), stored per admission: age (anchor_age + admission year − anchor_year), gender, admission type, admission location, admit/discharge times, length of stay.

Deterministic IDs: sort units by type order (diagnosis, procedure, lab, medication, history_diagnosis), then timestamp, then source key; assign E1…En. Two builds must produce identical files (tested).

Outputs: `data/derived/units/<hadm_id>.jsonl`, `data/derived/units/<hadm_id>_header.json`, and a unit-count summary table.

**GATE 2 — report:** unit counts by type per admission; 3 full example units of each type drawn from different admissions; one full header; anything dropped and why (null values, unparseable doses, etc.). Wait.

### Phase 3 — Retrieval

- Retrieval is always restricted to the units of ONE target admission. It must be impossible to return another admission's unit (tested).
- **Sparse:** `rank_bm25.BM25Okapi` over unit text; lowercase tokenization that keeps numbers intact.
- **Dense:** MedCPT. Queries encoded with `ncbi/MedCPT-Query-Encoder`, units with `ncbi/MedCPT-Article-Encoder`. L2-normalized embeddings, FAISS `IndexFlatIP` per admission. Cache embeddings on disk keyed by hash(unit text, model name).
- **Fusion:** Reciprocal Rank Fusion with `rrf_k` from config (default 60). Dedupe by unit_id.
- **Rerank:** optional `ncbi/MedCPT-Cross-Encoder`, off by default (`rerank: false`). Implement only after everything else in this phase works.
- **Queries:** per-section templates from config; `{primary_dx}` is filled with the long_title of the seq_num = 1 diagnosis. For each query take `top_k_per_query`; union per section (cap `max_units_per_section`); union across sections, deduped, cap `max_units_total`. Evidence order = first appearance in section order History → Findings → Impression → Recommendations.
- **Relevance export:** for `retrieval.relevance_export_pairs` (admission, query) pairs chosen with a fixed seed, write `runs/<id>/relevance_to_label.csv`: the pooled, deduped top-10 from BM25, dense, and fused, with columns hadm_id, query, unit_id, unit_text, relevant (blank). A `retrieval-metrics` command computes Recall@5 and MRR per method from my labeled CSV. State in its output that relevance is pooled, so recall is relative to the pool.
- **Acceptance:** tests pass; print retrieved units for 2 admissions × 4 sections.

### Phase 4 — Prompt and generation

Conditions (config `generation.conditions`):
- `no_retrieval`: header only, with an instruction not to cite anything. Baseline for what the model invents.
- `ehr_rag`: header + retrieved units from Phase 3.
- `full_context`: header + all units, only for admissions whose evidence fits `max_context_tokens`. Log skipped admissions.

Prompt template in `prompts/report_v1.txt`. Never edit a released version in place; create `report_v2.txt` instead. Required instruction content:
- Draft a clinical report for one hospital admission using ONLY the patient header and the evidence below.
- Output exactly four section header lines: `### History`, `### Findings`, `### Impression`, `### Recommendations`.
- One sentence per line. No bullets, no blank lines inside a section.
- Every sentence ends with citations in the form `[E3]` or `[E3, E7]`, using only IDs from the evidence block. Cite every unit the sentence relies on.
- Do not state patient-specific facts that are not in the evidence. If the evidence for a section is insufficient, write the single line `Insufficient evidence. [NONE]`.
- Impression and Recommendations may contain clinical reasoning but must cite the units the reasoning is based on.
- Evidence block: one unit per line, `[E12] (lab) <text>`, in retrieval order.

Backends (`generation/backends.py`), chosen by `generation.backend`:
- `anthropic` (Anthropic SDK).
- `openai_compatible` (OpenAI SDK with configurable `base_url`; covers hosted APIs and local servers such as vLLM or Ollama).

Generation rules:
- `generation.model` has no default. Refuse to run if unset.
- API key read from the environment variable named in `generation.api_key_env`. Never from files in the repo.
- Temperature 0 by default. Record the model id returned by the API.
- Response cache in `cache/`, key = sha256(prompt, model, params, sample_idx). `--no-cache` bypasses it.
- Before any generation run, print the number of calls and estimated input tokens; require `--yes` if calls exceed `confirm_threshold`.
- Retry transient errors with exponential backoff. On persistent failure, stop the run and report. No silently partial results.
- Format failure: one retry with a format reminder; if still invalid, mark `format_error` and keep the raw output.

Outputs per run: `runs/<id>/<condition>/<hadm_id>/prompt.txt`, `raw.txt`, `parsed.json`, `evidence_shown.json`. `evidence_shown.json` is the exact ordered list of unit IDs and texts the model saw; it is the source of truth for all evaluation.

**GATE 3 — report:** two full `ehr_rag` reports and one `no_retrieval` report, each with its evidence block; format-error counts per condition. Wait.

### Phase 5 — Evaluation harness

**Parser** (`eval/parse.py`): split by the exact section headers; each line is one sentence; extract the trailing citation group with a regex supporting `[E3]`, `[E3, E7]`, `[E3][E7]`, and `[NONE]`. Store sentence text without citations plus the ID list. Flag missing/extra sections, citations appearing mid-sentence, and lines that look like multiple sentences.

**Metrics** (per report, per section, and aggregated per condition):
- `cited_rate`: fraction of sentences with ≥1 valid citation.
- `invalid_citation_rate`: fraction of citations whose ID is not in `evidence_shown.json` (fabricated IDs).
- `citation_recall` (ALCE-style, Gao et al. 2023): a sentence is supported if NLI(premise = concatenated texts of its cited units, hypothesis = sentence) is entailment.
- `citation_precision` (ALCE-style): a citation is irrelevant if (a) it alone does not entail the sentence AND (b) removing it does not change whether the remaining citations entail the sentence. Precision = fraction of citations not irrelevant.
- Document in the README exactly what was implemented.
- Report History+Findings separately from Impression+Recommendations. Entailment is expected to be less meaningful for inferential sections.

**NLI model:** from `eval.nli_model`. Read the label mapping from the model's own config (`id2label`); never assume label order. If the model fails to load, STOP.

**Text metrics** (`eval/text_metrics.py`): ROUGE-L and BERTScore functions that REQUIRE a reference file. The Demo has no reference reports, so the command must exit with a clear message when none is provided. Never generate references.

**Manual review export:** `manual_review.csv` with `eval.manual_review_n` sentences sampled with a fixed seed, stratified by condition and section. Columns: run_id, condition, hadm_id, section, sentence, cited_ids, cited_texts, nli_label, and blank human_support (supported / partial / unsupported / not_checkable), human_error_type, notes.

**Verifier agreement:** `verifier-agreement` command computes raw agreement, Cohen's kappa, and a confusion matrix between NLI labels and my labels.

Outputs: `metrics.json`, `metrics_by_section.csv`, `metrics_summary.md`.

**GATE 4 — report:** metrics table for all conditions; 10 randomly sampled sentences shown side by side with their parsed citation IDs, the cited unit texts, and the NLI label, so I can check the harness by hand. Wait.

### Phase 6 — Perturbation pilot

**Frozen retrieval:** every perturbed or control run reuses exactly the ordered unit list in the original `ehr_rag` run's `evidence_shown.json`. Only the edited unit's fields (and therefore its rendered text) change. Never re-run retrieval for perturbation.

**Target selection** (`perturb/select_targets.py`): from the original `ehr_rag` reports, find sentences that cite a lab or medication unit AND contain a number matching a value in that unit's fields. Numeric matching is tolerant of formatting (5.9 == 5.90) but must not match inside larger numbers (5.9 must not match 15.9). Save all candidates, then choose up to `perturbation.n_targets`, at most `max_targets_per_admission` per admission, with a fixed seed. Save `targets.csv`.

**Edit types** (`perturb/edit.py`), all operating on `fields` followed by re-rendering:
- `lab_minor`: change the matched value by 10–20% while staying on the same side of the reference range (same HIGH/LOW/normal status).
- `lab_major`: move the matched value to a different status (normal → clearly abnormal, e.g. 1.5× upper limit or 0.7× lower limit; abnormal → middle of the normal range). Markers update accordingly.
- `med_dose`: double or halve `dose_val_rx`.
- `delete`: remove the unit from the evidence block. Other unit IDs are NOT renumbered.

Edited values keep the original's decimal precision. If an edit would make the unit internally inconsistent (e.g. new max below the last value), skip that target and log it. Log original fields, edited fields, original text, and edited text for every edit.

**Runs:** each target × each edit type gets one regeneration at temperature 0. Controls: each target gets `n_controls` unperturbed regenerations with identical evidence, using distinct `sample_idx` values so the cache does not collapse them.

**Outcome labeling** (`perturb/analyze.py`). Candidate sentences in a regenerated report = sentences citing the target unit ID, plus sentences mentioning the unit's lab label or drug name (case-insensitive). Classify each case:
- `followed`: a sentence states the edited value and cites the target unit.
- `updated_miscited`: a sentence states the edited value but does not cite the target unit.
- `stale`: a sentence still states the original value.
- `dropped`: no sentence mentions the fact.
- `ambiguous`: anything else (goes to manual review).

For `delete`: `abstained` (fact not mentioned) vs `stale` (original value still stated), and separately whether any sentence still cites the deleted ID.

Also record whether the Impression or Recommendations text changed relative to the original (store a diff; flag `lab_major` cases for manual reading). Control stability = fraction of control runs in which the target sentence is unchanged.

Outputs: `perturbation_results.csv` (one row per case), `perturbation_summary.md` (rates per edit type, always shown with raw counts, e.g. `7/12`), and `perturbation_manual_check.csv` containing every case. Automated labels are provisional until I check them.

**GATE 5 — report:** the summary table, control stability, and 3 worked examples per edit type (original sentence → edit → regenerated sentence, with citations). Wait.

### Phase 7 — Demo notebook and README

- `notebooks/demo.ipynb`: one admission end to end: header, sample units, retrieval per section, prompt, rendered report with citations, citation check results, one perturbation example with a diff. Runs top to bottom from cached outputs; no new API calls by default.
- README: setup, data download, every CLI command in order, config fields, metric definitions exactly as implemented, known limitations (Demo data only, no clinical notes, pooled relevance judgments, NLI verifier limitations, small N).

---

## 5. Compliance guard

If `data.source == "full"` (credentialed MIMIC-IV) and the backend points at any external API, refuse to run unless `generation.endpoint_is_physionet_compliant: true`, and print a warning that credentialed data may only be sent to services permitted under the PhysioNet data use agreement. `demo` data is open access and not subject to this guard.

---

## 6. Required tests

- Parser: single citation, multiple, adjacent brackets, `[NONE]`, missing section, malformed lines.
- Unit IDs identical across two independent builds.
- `render` is pure; editing one unit's fields and re-rendering changes only that unit.
- Retrieval isolation: never returns a unit from another admission.
- Perturbation: markers update when a value crosses the reference range; `delete` does not renumber; frozen evidence order is preserved.
- Numeric matcher: matches 5.9 / 5.90 / "5.9 mmol/L"; does not match 15.9 or 5.95.
- `no_retrieval` prompt contains no unit text.
- Config validation rejects unknown keys and an unset `generation.model`.

---

## 7. Out of scope (do not build)

Fine-tuning of any kind; a literature/guideline corpus (keep the retriever interface generic so one can be added later, but do not build it); synthetic clinical notes; the conflicting-records probe; ICU chartevents/vitals; microbiology units; imaging; any web UI; orchestration frameworks; ROUGE/BERTScore against fabricated references.

---

## 8. Default config (`config/default.yaml`)

```yaml
data:
  source: demo                 # demo | full
  root: data/mimic-iv-demo
  derived: data/derived

selection:
  min_distinct_labs: 10
  min_abnormal_labs: 3
  min_distinct_drugs: 5
  min_diagnoses: 3
  n_admissions: 15

units:
  max_prior_admissions: 3
  max_history_dx_per_admission: 5

retrieval:
  dense_query_model: ncbi/MedCPT-Query-Encoder
  dense_doc_model: ncbi/MedCPT-Article-Encoder
  rrf_k: 60
  top_k_per_query: 5
  max_units_per_section: 10
  max_units_total: 30
  rerank: false
  rerank_model: ncbi/MedCPT-Cross-Encoder
  relevance_export_pairs: 10
  queries:
    History:
      - "past medical history and prior diagnoses"
      - "chronic conditions related to {primary_dx}"
    Findings:
      - "abnormal laboratory results"
      - "laboratory values related to {primary_dx}"
      - "procedures performed during admission"
    Impression:
      - "primary diagnosis {primary_dx}"
      - "secondary diagnoses and complications"
    Recommendations:
      - "medications administered"
      - "treatment for {primary_dx}"

generation:
  backend: null                # anthropic | openai_compatible  (REQUIRED)
  model: null                  # REQUIRED
  base_url: null
  api_key_env: null            # name of the environment variable holding the key
  temperature: 0.0
  max_tokens: 1500
  prompt_version: report_v1
  conditions: [no_retrieval, ehr_rag, full_context]
  max_context_tokens: 12000
  confirm_threshold: 20
  endpoint_is_physionet_compliant: false

eval:
  nli_model: cross-encoder/nli-deberta-v3-base   # verify it loads; read id2label
  manual_review_n: 100
  seed: 13

perturbation:
  n_targets: 15
  max_targets_per_admission: 2
  edit_types: [lab_minor, lab_major, med_dose, delete]
  n_controls: 3
  seed: 13
```
