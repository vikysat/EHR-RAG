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
