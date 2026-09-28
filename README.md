# EHR-RAG

EHR-aware clinical RAG with citation-faithfulness probes. See `SPEC.md` for the
full specification. This README is a stub and is completed in Phase 7.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.11+ for the pinned dependencies (pandas 3.x); the code itself
avoids syntax newer than 3.10.

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
server too.

## Data

Download the MIMIC-IV Clinical Database Demo (v2.2, open access) from PhysioNet
and point `data.root` in `config/default.yaml` at the extracted directory.

## Usage

```bash
python -m ehr_rag.cli --help
```

### Phase 4: generation

Backends (`generation.backend`): `hf_local` (default; Hugging Face transformers,
in-process, bf16 on CUDA, the model's chat template, greedy decoding),
`anthropic`, `openai_compatible`. API backends read the key from the environment
variable named in `generation.api_key_env`; `hf_local` needs no key and refuses to
run without CUDA. The `anthropic` backend cannot set a temperature (the pinned SDK
and current Claude models do not accept one); `call.json` records
`temperature_applied: false` for it.

`ehr_rag` reuses the ordered evidence saved by a `retrieve` run (never re-retrieved),
so run retrieval first and pass its run id:

```bash
python -m ehr_rag.cli retrieve                     # -> runs/<retrieve_run_id>/retrieval/<hadm_id>.json
python -m ehr_rag.cli context-tokens               # full_context prompt tokens vs generation.max_context_tokens
python -m ehr_rag.cli generate --retrieval-run <retrieve_run_id> [--conditions ...] [--limit N] [--yes] [--no-cache]
```

`generate` prints the number of calls and estimated input tokens first and requires
`--yes` when new (uncached) calls exceed `generation.confirm_threshold`. Responses are
cached in `cache/generation/`. Output per condition and admission:
`runs/<id>/<condition>/<hadm_id>/{prompt.txt, raw.txt, parsed.json, evidence_shown.json, call.json}`
(plus `*_retry.txt` after a format retry and `prompt_templated.txt` for `hf_local`).

On the SLURM GPU server (submit from the repo root; the model must already be in `HF_HOME`,
the job forces `HF_HUB_OFFLINE=1`):

```bash
mkdir -p runs
sbatch scripts/generate.sbatch --retrieval-run <retrieve_run_id> --conditions ehr_rag --limit 2   # smoke test
sbatch scripts/generate.sbatch --retrieval-run <retrieve_run_id> --yes                            # all conditions
```

The job writes `nvidia-smi.txt`, `slurm_env.txt`, `stdout.log` and `stderr.log` into the
run folder; SLURM's own log goes to `runs/slurm-<jobid>.out`.
