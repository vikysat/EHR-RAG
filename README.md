# EHR-RAG

EHR-aware clinical RAG with citation-faithfulness probes, on the MIMIC-IV Clinical
Database Demo (structured data only).

- **Input:** one hospital admission.
- **Output:** a report with four sections (History, Findings, Impression,
  Recommendations) in which every sentence cites evidence-unit ids such as `[E12]`.
- **Question:** are the citations faithful? The test is counterfactual. Edit one cited
  fact, regenerate with retrieval frozen, and check whether the claim and its citation
  follow the edit.

Companion documents:
- [`RUNBOOK.md`](RUNBOOK.md): exact server commands, in order, with the checks for
  each gate.
- [`DECISIONS.md`](DECISIONS.md): every judgment call where the specification was
  ambiguous.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m pytest                      # all tests use fakes; no models are downloaded
```

Requires Python 3.11+ for the pinned dependencies (pandas 3.x); the code itself
avoids syntax newer than 3.10. Jupyter is not a dependency; install it separately
(`pip install jupyterlab`) to open `notebooks/demo.ipynb`.

### GPU server (CUDA 12.x)

The server's NVIDIA driver supports CUDA 12.x only, so torch **must** be installed
from the CUDA 12.6 index `https://download.pytorch.org/whl/cu126`, **before** the
rest of the requirements (otherwise pip installs a different torch build from PyPI):

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt   # torch==2.14.0 is already satisfied and is not replaced
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# expected: 2.14.0+cu126 12.6 True
```

Do not use the cu128/cu13x indexes on this server. faiss stays `faiss-cpu` on the
server too. GPU jobs run with `HF_HUB_OFFLINE=1`, so download every model into
`HF_HOME` on the login node first (RUNBOOK step 0).

## Data

Download the MIMIC-IV Clinical Database Demo v2.2 (open access) from PhysioNet
(<https://physionet.org/content/mimic-iv-demo/2.2/>, ZIP download). Extract it so
that `data/mimic-iv-clinical-database-demo-2.2/hosp/` exists; that directory is
`data.root` in `config/default.yaml` (point it elsewhere if needed). MIMIC dates are
shifted into the 2100s on purpose. `data/`, `cache/` and `runs/` are gitignored and
never committed.

## Commands, in order

All commands are `python -m ehr_rag.cli <command> [--config config/default.yaml] ...`.
Every command that produces results writes `runs/<YYYYMMDD-HHMMSS-name>/`, containing:
- `config.yaml` (the resolved config);
- `run_meta.json` (command, argv, seed, timestamps, git commit, package versions,
  model ids and revisions, status);
- the command's outputs.

| # | command | where | what it does / main outputs |
|---|---|---|---|
| 0 | `validate-config` | any | validate and print the config (unknown keys are errors) |
| 1 | `inspect-data` | CPU | data tree, header check vs SPEC, 5 rows per table, row counts, lab linkage |
| 2 | `select-admissions` | CPU | admission metrics, `data/derived/admission_candidates.csv`, proposed `config/selected_admissions.yaml` (edit by hand) |
| 2a | `select-cohort [--skip-note-filter]` | CPU | full MIMIC-IV (DuckDB over csv.gz): seeded dev/test cohort; prints counts only; id lists in `cohort.out_dir` (gitignored), SHA-256 hashes in `config/cohort/*.sha256` |
| 2b | `extract-cohort [--splits dev test]` | CPU | copy the cohort patients' rows of the 9 builder tables (Demo columns, as strings) into `<cohort.out_dir>/evidence/hosp/*.parquet`, filtering in DuckDB; their discharge notes into `<cohort.out_dir>/reference/` (`discharge.parquet`, `references.csv`); counts only |
| 2c | `cohort-schema-check` | CPU | headers only: columns that differ between full MIMIC-IV, the Demo and the extracted Parquet; exit 3 if a Demo or builder column is missing |
| 3 | `build-units` | CPU | evidence units + headers: `data/derived/units/<hadm_id>.jsonl`, `<hadm_id>_header.json` |
| 4 | `retrieve` | CPU/GPU | BM25 + MedCPT dense + RRF per section; `retrieval/<hadm_id>.json` incl. the ordered `evidence_units` used by `generate` |
| 5 | `export-relevance` | CPU/GPU | `relevance_to_label.csv`: pooled top-10 (BM25, dense, fused) for manual labeling |
| 6 | `retrieval-metrics --labels <csv>` | CPU | Recall@5 and MRR per method from your labels (relative to the pool) |
| 7 | `context-tokens` | CPU | full_context templated-prompt tokens per admission vs `max_context_tokens` |
| 8 | `generate --retrieval-run <id>` | GPU (`scripts/generate.sbatch`) | reports for no_retrieval / ehr_rag / full_context: `<condition>/<hadm_id>/{prompt.txt, raw.txt, parsed.json, evidence_shown.json, call.json}` |
| 8b | `generate ... --shard-size 25 --shard-index <k>` | GPU (`scripts/generate_array.sbatch`, one array task per shard) | contiguous blocks of the admission list; each shard is its own run `...-generate-job<array id>-shard<k>` |
| 8c | `generate --conditions ehr_rag ehr_rag_enforced ...` | GPU | `*_enforced`: the base report, then each line failing the citation checker is rewritten once by the same model and re-checked, or dropped: `enforced.txt`, `enforcement.json`, `rewrites/` |
| 8d | `enforce-filter --generation-run <id...>` | CPU | no model: copies ehr_rag / full_context reports to `*_filtered` and drops failing lines |
| 8e | `check-citations --generation-run <id...>` | CPU | checker statistics only: status and coverage, error profile per section and condition, repair vs retreat, `checker_review.csv` for tuning the lexicon |
| 9 | `evaluate --generation-run <id...>` | GPU/CPU (`scripts/cli.sbatch`) | `metrics.json`, `metrics_by_section.csv`, `metrics_summary.md`, `sentences.jsonl`, `manual_review.csv`, 10-sentence check |
| 10 | `verifier-agreement --labels <csv>` | CPU | raw agreement, Cohen's kappa, confusion matrix (NLI vs your labels) |
| 11 | `text-metrics --generation-run <id...> --references <csv> [--bertscore]` | CPU/GPU (`scripts/cli.sbatch`) | ROUGE-L / windowed BERTScore of each report vs the admission's discharge summary; exits without a reference file (the Demo has none) |
| 12 | `select-targets --generation-run <id>` | CPU | `candidates.csv`, `targets.csv`, `edits.jsonl`, `skipped_edits.csv` |
| 13 | `run-pilot --targets-run <id>` | GPU (`scripts/cli.sbatch`) | frozen-evidence regenerations per edit + controls: `cases/<case_id>/...`, `cases.csv` |
| 14 | `analyze-perturbation --pilot-run <id>` | CPU | `perturbation_results.csv`, `perturbation_summary.md`, `perturbation_manual_check.csv`, `diffs/` |
| 14a | `audit-targets --generation-run <id...>` | CPU | counterfactual audit targets: per admission up to 3 cited labs (flagged first) + 1 uncited null-edit lab; edits to the other side of the reference range: `audit_targets.csv`, `audit_edits.jsonl`, `audit_skips.csv` |
| 14b | `run-audit --targets-run <id>` | GPU (`scripts/audit_array.sbatch`) | regenerations with the edited unit in the frozen evidence, for `ehr_rag` and `ehr_rag_enforced` (frozen checker + rewrite); shardable over admissions |
| 14c | `analyze-audit --audit-run <id...>` | CPU | Follows / Stale / Dropped / Other per (original claim, edited unit); faithfulness with 95% bootstrap CIs (paired plain vs enforced); term survival; null-edit churn; `audit_manual_check.csv` (50 pairs) |
| 16 | `make-tables [--evaluate-run] [--manual-labels] [--check-run] [--text-metrics-run] [--audit-run]` | CPU | the paper's tables as markdown with 95% CIs: `tables.md`, `table_*.md` |
| 15 | `notebooks/demo.ipynb` | any | one admission end to end from cached runs; no model calls |

`generate` and `run-pilot` print the number of calls and the estimated input tokens
first. They require `--yes` when the new (uncached) calls exceed
`generation.confirm_threshold`. `--no-cache` bypasses the response cache in
`cache/generation/`. GPU commands take `--run-id`, which the sbatch scripts set; the
scripts also save `nvidia-smi.txt`, `stdout.log` and `stderr.log` in the run folder.

### Generation backends

`generation.backend`:
- `hf_local` (default): Hugging Face transformers in-process, bf16 on CUDA, the
  model's chat template, greedy decoding (`do_sample=False`), and the model's own
  `eos_token_id` list. Refuses to run without CUDA and records the model's revision
  hash.
- `anthropic` and `openai_compatible` (OpenAI SDK with `base_url`): read the key from
  the environment variable named in `api_key_env`, and retry transient errors with
  exponential backoff. `anthropic` cannot set a temperature (see DECISIONS.md).

Outputs that failed the format check get one retry with a format reminder; if the
retry also fails, the report is marked `format_error` and the raw output is kept.

### Full MIMIC-IV (`config/full.yaml`)

`config/full.yaml` is `default.yaml` with a different `data` block: `data.source: full`,
`data.root: data/cohort/evidence` (the Parquet from `extract-cohort`) and
`data.derived: data/derived_full`. The builder code is unchanged. `build-units`,
`retrieve`, `export-relevance`, `generate` and `context-tokens` take `--split dev|test`,
which reads the cohort id list and checks it against `config/cohort/<split>_hadm_ids.sha256`.
With `data.source: full`, commands print counts only; examples (units, retrieval
sections, the 10-sentence check, perturbation examples) stay in the run folder.

The current admission's discharge summary never enters the evidence: `load_tables`
refuses a data root that contains any note table (`discharge*`, `radiology*`, or a
`note/` directory), and `build-units` and `generate` refuse units whose
`source_table` is not one of diagnoses_icd, procedures_icd, labevents, prescriptions.
Discharge notes are extracted only to `<cohort.out_dir>/reference/`, for `text-metrics`.

## Configuration (`config/default.yaml`)

Unknown or missing keys are errors.

| key | meaning |
|---|---|
| `data.source` | `demo` or `full`. With `full` and an external backend, runs are refused unless `generation.endpoint_is_physionet_compliant: true` |
| `data.root` | extracted MIMIC-IV directory |
| `data.derived` | derived files (candidates, units) |
| `selection.min_distinct_labs / min_abnormal_labs / min_distinct_drugs / min_diagnoses` | eligibility thresholds |
| `selection.n_admissions` | number of proposed target admissions |
| `cohort.mimic_root / note_root / demo_root` | full MIMIC-IV (`hosp/`), MIMIC-IV-Note (`note/`), Demo (subject ids to exclude, reference columns) |
| `cohort.out_dir` | gitignored: `<split>_hadm_ids.txt`, extracted Parquet, reference notes |
| `cohort.pool_fraction / seed` | seeded share of all subject_ids drawn before the filters |
| `cohort.min_los_days / max_los_days / min_prior_admissions / min_lab_results` | eligibility (inclusive) |
| `cohort.n_dev / n_test` | admissions per split (one per patient; splits share no patient) |
| `units.max_prior_admissions`, `units.max_history_dx_per_admission` | caps for history_diagnosis units |
| `retrieval.dense_query_model / dense_doc_model` | MedCPT encoders |
| `retrieval.rrf_k` | Reciprocal Rank Fusion constant |
| `retrieval.top_k_per_query / max_units_per_section / max_units_total` | evidence assembly caps |
| `retrieval.rerank`, `retrieval.rerank_model` | optional MedCPT cross-encoder rerank (off) |
| `retrieval.relevance_export_pairs` | (admission, query) pairs exported for labeling |
| `retrieval.queries` | per-section query templates; `{primary_dx}` = seq_num 1 diagnosis |
| `generation.backend`, `generation.model` | required; default `hf_local` + `meta-llama/Llama-3.1-8B-Instruct` |
| `generation.base_url`, `generation.api_key_env` | API backends only |
| `generation.temperature` | 0 (must be 0 for `hf_local`) |
| `generation.max_tokens` | max new tokens |
| `generation.prompt_version` | `prompts/<version>.txt` (+ `_no_retrieval`, `_format_reminder`); released versions are never edited |
| `generation.conditions` | subset of no_retrieval, ehr_rag, full_context |
| `generation.max_context_tokens` | full_context admissions whose whole templated prompt is longer are skipped (logged) |
| `generation.confirm_threshold` | new calls above this need `--yes` |
| `generation.endpoint_is_physionet_compliant` | see `data.source` |
| `eval.nli_model` | NLI verifier (label order read from its config) |
| `eval.manual_review_n`, `eval.seed` | manual-review sample size and seed |
| `perturbation.n_targets / max_targets_per_admission / seed` | target selection |
| `perturbation.edit_types` | subset of lab_minor, lab_major, med_dose, delete |
| `perturbation.n_controls` | unperturbed regenerations per target |

## Citation checker and enforcement

A deterministic checker tests every generated line against ONLY the units it cites
(rules: `config/checker.yaml`, vocabulary: `config/checker_lexicon.yaml`, definitions:
`src/ehr_rag/checker/rules.py`). Error types: `invalid_id`, `number_mismatch`,
`name_missing`, `direction_mismatch`, `trend_mismatch`, `too_many_citations`. Each line is
`pass`, `fail` or `unchecked` (no checkable number, name, direction or trend); unchecked
lines are reported separately as the checker's coverage. List lines are checked per item,
each item keeping the list's opening words.

The checker lives in `ehr_rag.checker`, which `ehr_rag.eval` never imports: `evaluate`
grades enforced reports with NLI and manual labels only, and checker statistics come
only from `check-citations`, in their own files. The SHA-256 of the rules, the lexicon and
the rewrite prompt is in every run's `run_meta.json` (`checker`) and printed by
`check-citations`, so the frozen versions used on dev and test can be compared.

## Metric definitions (as implemented)

**Parsing** (`eval/parse.py`):
- Sections are split on the exact lines `### History`, `### Findings`,
  `### Impression`, `### Recommendations`. Every other non-blank line in a section is
  one sentence.
- The trailing citation group (`[E3]`, `[E3, E7]`, `[E3][E7]`, `[NONE]`, optionally
  followed by one period: `Fact [E3].`) gives the line's citations. Ids are
  de-duplicated. The generation format check accepts the same trailing period.
- Citations elsewhere in a line are flagged `mid_sentence_citation`. Lines with a
  sentence boundary (`.`/`!`/`?` + space + capital letter) are flagged
  `multiple_sentences`.
- Missing, duplicate, extra and out-of-order sections, and text before the first
  header, are flagged per report.

**Claims.** Before scoring, a line with citations inside it (a list such as
`Na 130 [E1], K 5.9 [E2], and Cr 2.1 [E3].`) is split into one claim per citation
run, each scored with only its own ids (`Na 130` [E1], `K 5.9` [E2], `Cr 2.1` [E3]).
Leading `,;:`/`and`/`or` are removed; text after the last citation is an uncited
claim. Other lines are one claim. Claims of a split line have `sentence_key`
`...|<line>.<k>` and carry the whole line (`full_line` in `manual_review.csv`).

**Metrics** (`eval/citations.py`) use every claim of the four sections except
abstention lines `Insufficient evidence. [NONE]` (reported as `n_abstentions`).
`n_sentences` counts lines, `n_claims` claims; rates are over claims. A citation is
*valid* if its id is in the report's `evidence_shown.json`. The NLI premise is only
the texts of the claim's valid cited units, joined by a space in citation order.

- `cited_rate`: claims with at least one valid citation / claims.
- `invalid_citation_rate`: E-id citations not in `evidence_shown.json` / all E-id
  citations (`[NONE]` is not a citation).
- `citation_recall` (ALCE, Gao et al. 2023): a claim is supported iff it has at
  least one valid citation and NLI(premise, hypothesis = claim) is `entailment`.
  Recall = supported / claims.
- `citation_precision` (ALCE): a valid citation c of a claim with valid set C is
  irrelevant iff NLI(c alone) is not entailment AND entail(C − {c}) == entail(C),
  with entail(∅) = false. Invalid ids are irrelevant. Precision = relevant citations /
  all E-id citations.
- `citation_precision_plain`: citations that entail their claim on their own / all
  E-id citations (invalid ids never do). Unlike ALCE, a citation that is needed only
  jointly with another counts as not supporting.
- Aggregates are micro-averages per condition, per section, and for History+Findings
  vs Impression+Recommendations (entailment is less meaningful for the inferential
  sections). A zero denominator gives n/a.
- NLI model: `eval.nli_model` (default `cross-encoder/nli-deberta-v3-base`). The
  label order comes from the model's `config.id2label`. Nothing is truncated: a pair
  longer than 512 tokens is not run, gets the label `too_long`, and counts as not
  entailed. Every such pair is listed in `nli_too_long.csv` (claim, premise ids,
  token count) and counted in `metrics.json` (`nli.n_too_long_pairs`).
- Reports that ended in `format_error` are included (best-effort parse of the final
  raw output) and counted per condition.

**Manual review** (`manual_review.csv`): `eval.manual_review_n` sentences,
abstentions excluded, equal allocation over (condition, section) strata, seeded with
`eval.seed`. Fill `human_support` with supported / partial / unsupported /
not_checkable. `verifier-agreement` binarizes the labels (NLI entailment vs human
`supported`; partial and unsupported count as not supported; not_checkable is
excluded) and reports raw agreement, Cohen's kappa and the full confusion matrix.

**ROUGE-L / BERTScore** (`text-metrics`) score each generated report against the
same admission's real discharge summary (`<cohort.out_dir>/reference/references.csv`
from `extract-cohort`, columns `hadm_id,reference`). Without a reference file the
command exits; references are never generated. Reports are scored as their sentence
text only (no headers, citations or abstention lines).
- ROUGE-L: longest common subsequence of lowercased word tokens; precision, recall, F1.
- BERTScore: greedy cosine matching of contextual token embeddings (no idf, no baseline
  rescaling), with `eval.bertscore_model` (default `roberta-large`) layer
  `eval.bertscore_layer` (17). Discharge summaries are far longer than 512 tokens, so
  each text is embedded in consecutive 510-token windows and all token embeddings are
  matched; nothing is truncated. Implemented in-repo with transformers (no bert-score
  package), loaded with `local_files_only=True`. Values are not identical to the
  bert-score package's, which truncates at 512 tokens.
- Output: `text_metrics.csv` per (condition, admission) in the run folder; the
  command prints only counts and per-condition mean/median F1.

## Perturbation (as implemented)

- **Targets:** sentences in the original ehr_rag reports whose trailing citations
  include a numeric lab unit or a medication with a numeric dose, and that state a
  number equal to one of that unit's values. The matcher tolerates formatting (5.9 ==
  5.90) and never matches inside larger numbers (15.9, 5.95), dates or ids. Up to
  `n_targets`, at most `max_targets_per_admission` per admission, chosen with a fixed
  seed.
- **Edits,** always on `fields` followed by `render()`:
  - `lab_minor`: ±10–20 %, same HIGH/LOW/normal status.
  - `lab_major`: normal → 1.5 × upper limit (or 0.7 × lower limit); abnormal → middle
    of the normal range.
  - `med_dose`: doubled.
  - `delete`: unit removed, no renumbering.

  Decimal precision is kept. Edits that would make a unit inconsistent are skipped
  and logged.
- **Frozen retrieval:** each regeneration uses the original `evidence_shown.json`
  order. The unperturbed prompt must rebuild byte-identical to the original
  `prompt.txt`, or the pilot stops. Controls: `n_controls` unperturbed regenerations
  per target with distinct `sample_idx`.
- **Outcomes** (provisional until checked by hand):
  - value edits: followed, updated_miscited, stale, dropped, ambiguous;
  - delete: abstained, stale, ambiguous, plus whether the deleted id is still cited.

  Candidate sentences are those citing the target unit or naming its lab label or
  drug. Changes to Impression/Recommendations are stored as diffs; all lab_major and
  ambiguous cases are flagged for manual reading. Control stability is the fraction
  of control runs in which the original target sentence is unchanged.

## Known limitations

- **Demo data only:** 100 patients, and 15 selected admissions. All numbers are small-N
  and descriptive; there is no statistical testing.
- **Structured data only:** no clinical notes, vitals, microbiology or imaging. The
  reports can only restate coded diagnoses, procedures, labs and prescriptions.
- **Pooled relevance judgments:** Recall@5 is relative to the pool of the three
  retrievers' top-10, not to all relevant units.
- **NLI verifier limitations:**
  - It is a general-domain sentence-pair model (`cross-encoder/nli-deberta-v3-base`),
    not trained on clinical text, numbers or units.
  - Pairs longer than 512 tokens are labeled `too_long` and count as not entailed,
    which lowers recall for claims that cite many long units (see `nli_too_long.csv`).
  - Entailment is not a good test for the inferential Impression/Recommendations
    sentences, which is why they are reported separately.
  - Agreement with human labels has to be measured (`verifier-agreement`) before the
    metrics are trusted.
- **One model, greedy decoding:** results are for Llama-3.1-8B-Instruct at temperature
  0. Under greedy decoding the control regenerations are expected to be identical,
  so control stability mostly checks determinism.
- **Heuristic outcome labels:** perturbation outcomes use number matching and
  label/drug-name matching. They are provisional and every case is exported for
  manual checking.
- **Tested on fakes only:** Phases 4–7 were built and tested on a CPU laptop against
  fake backends, fake NLI and fake data. The first real GPU runs (RUNBOOK) are the
  first end-to-end check on real outputs.
- **Reference reports only on full MIMIC-IV:** ROUGE-L and BERTScore need MIMIC-IV-Note
  discharge summaries, so they are never computed on the Demo; they were tested only
  on synthetic text and a tiny random encoder. A discharge summary also contains far
  more than the structured evidence can support (narrative, follow-up), so low
  overlap is expected for every condition.
