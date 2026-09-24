# CLAUDE.md

Follow SPEC.md exactly. The operating rules in SPEC.md §0 are non-negotiable.
Stop at every GATE and wait for my reply "continue".

## Current scope
Phases 0–3 only. After Phase 3's acceptance output, STOP. Do not start Phase 4.

## Environment
- Running on a Windows laptop, CPU only. No GPU.
- Python virtual environment at `.venv/` (add `.venv/` to .gitignore).
- Install the standard (CUDA-enabled) torch build, not a CPU-only build, so the same
  requirements work on a GPU server later.
- All paths come from config. Never hardcode laptop-specific paths.
- Use pathlib for all file paths so code runs on both Windows and Linux.

## Data safety
- Never commit anything under data/, cache/, runs/, or .venv/.
- If data.source is ever "full", never open, print, or read any file under
  data/, cache/, or runs/.
