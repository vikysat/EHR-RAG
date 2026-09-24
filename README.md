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

On a CUDA server, install torch from the PyTorch index **before** the rest:

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu128   # choose the cuXXX tag for your driver
pip install -r requirements.txt
```

## Data

Download the MIMIC-IV Clinical Database Demo (v2.2, open access) from PhysioNet
and point `data.root` in `config/default.yaml` at the extracted directory.

## Usage

```bash
python -m ehr_rag.cli --help
```
