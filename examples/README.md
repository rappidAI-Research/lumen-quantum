# Examples

Two small, runnable examples. Both avoid network access and neither downloads a
model or dataset.

## 1. Repository health quickstart (CPU-only, no ML dependencies)

```bash
python examples/repo_health_quickstart.py
```

Validates tracked configuration and report files, runs the repository-health,
secret, Markdown-link and canonical-URL checks, and writes a tiny synthetic
dataset to a temporary directory. This is the fastest way to confirm a fresh
clone is consistent.

## 2. Minimal ML pipeline (optional ML dependencies)

```bash
python -m pip install -e ".[ml]"
python examples/minimal_ml_pipeline.py
```

Trains a tiny SentencePiece tokenizer on self-generated text, builds a very
small randomly initialized Llama-style model, and runs a few training steps. It
is deliberately a **pipeline smoke test, not a usable language model**. Without
the optional dependencies it prints installation instructions and exits cleanly.

Runtimes are intentionally not stated because they have not been measured on a
reference machine.
