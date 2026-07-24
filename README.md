# rappidAI Quantum

Reproducible pipelines for preparing data, training tokenizers, developing,
evaluating, and exporting compact language models.

> **Experimental status:** this repository is pre-alpha research software. It
> does not provide a production model, completed Echelon training run, safety
> certification, benchmark leadership claim, or support guarantee.

## Purpose

rappidAI Quantum is the technical model-development repository for the rappidAI
research initiative. It contains Python code, versioned configurations, tests,
and reports for data preparation, tokenizer training, model construction,
checkpoint/resume, evaluation, and GGUF conversion. The separate website
repository is a publication surface, not the technical core.

Lumen was an earlier internal/project name and remains in historical commands,
paths, and source identifiers. Files, the GitHub repository, and public model
IDs are intentionally not renamed by this change.

## Capabilities

- deterministic splitting, filtering, manifests, and tokenization;
- SentencePiece tokenizer training and validation;
- compact Llama-style causal-decoder configuration and CPU smoke tests;
- random-from-scratch pilot training and weights-only continued pretraining;
- checkpoint, optimizer, scheduler, RNG, and resume handling;
- loss and fixed-prompt evaluation infrastructure;
- GGUF export through an explicitly supplied external llama.cpp checkout; and
- Echelon architecture, tokenizer, and Garden data-pipeline preflight reports.

## Model family

| Model line | Role | Verified public state |
|---|---|---|
| `quantum-1-pilot` | Independently pretrained pilot stage | Experimental F16 GGUF published; reuse terms remain unresolved |
| `quantum-1.6-pilot` | Continued-pretraining pilot stage | Experimental F16 GGUF published; publisher-reported metrics; final run manifest unavailable |
| `quantum-1-echelon` | Current strategic model line | Architecture/tokenizer/data preflight only; no trained checkpoint released |

Echelon Base and Echelon Chat are stages or variants inside
`quantum-1-echelon`, not separate model families. See
[`docs/model-lineage.md`](docs/model-lineage.md) and the canonical model cards.

## CPU-only quickstart

Python 3.11 and 3.12 are supported. This lightweight path does not install
PyTorch, access a dataset, download a model, or require a GPU.

```bash
git clone https://github.com/rappidAI-Research/lumen-quantum.git
cd lumen-quantum
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/validate_repository_configs.py
python scripts/repository_health.py
python -m pytest tests/test_repository_health.py -m unit
```

On PowerShell, activate with `.\.venv\Scripts\Activate.ps1`; the Python
commands are unchanged.

## Safe smoke test

The default contributor smoke test is offline and CPU-only:

```bash
ruff format --check .
ruff check .
mypy
python scripts/validate_repository_configs.py
python scripts/check_markdown_links.py
python scripts/check_secrets.py
python -m pytest -m "not slow and not gpu and not network"
```

The final test command requires the ML extra. Install it with
`python -m pip install -r requirements-dev.txt`. It can be substantially larger
than the lightweight setup and must not download model or dataset artifacts.

## Architecture

The repository is configuration-driven:

```text
versioned YAML + local/streamed inputs
  -> data filtering, stable splits, manifests
  -> tokenizer training and validation
  -> Llama-style model configuration and preflight
  -> CPU/GPU training with checkpoints and resume
  -> loss and completion evaluation
  -> explicit external llama.cpp GGUF conversion
```

Generated datasets, tokenizers, checkpoints, logs, and exports are ignored.
Only small evaluation prompts and evidence reports are tracked. See
[`docs/architecture.md`](docs/architecture.md).

## Data pipeline

Pilot and Echelon configurations reference `epfml/FineWeb2-HQ`. The Echelon
production config pins a dataset revision; several older pilot configs still use
mutable `main` and are not release-reproducible. Dataset terms, source-page
rights, personal-information handling, filtering, and removal obligations are
separate from the source-code license. See [`docs/datasets.md`](docs/datasets.md)
and [`DATA_SOURCES.md`](DATA_SOURCES.md).

## Tokenizer pipeline

The pilots and Echelon use separately versioned SentencePiece configurations.
Tokenizer identity is enforced with vocabulary, special-token, and checksum
checks where evidence exists. Tokenizer binaries are ignored and do not inherit
Apache-2.0. See [`docs/tokenizers.md`](docs/tokenizers.md).

## Training and resume

The historical pilot line starts from random weights. `quantum-1.6-pilot`
performs weights-only initialization from the earlier pilot while resetting the
optimizer, scheduler, and step. Resume files contain Python pickle data through
`torch.load(..., weights_only=False)` and must be treated as trusted local
artifacts only. See [`docs/training.md`](docs/training.md).

The historical “no pretrained weights” rule applies to the from-scratch pilot
design. Future fine-tuned models may use upstream weights only when the model ID,
exact revision, license, tokenizer, and modifications are documented.

## Evaluation

Evaluation code supports validation loss/perplexity and fixed completion
prompts. Current pilot metrics in the model cards are publisher-reported unless
linked to raw versioned output. No broad benchmark suite or statistical quality
claim is available. See [`docs/evaluation.md`](docs/evaluation.md).

## GGUF export

llama.cpp is not vendored or a submodule. Obtain an external checkout, verify
revision `d4cff114c0084f1fbc9b4c62717eca8fb2ae494a` for the previously tested
baseline, then supply its path explicitly:

```bash
python scripts/export_gguf.py \
  --model-dir models/smoke/final \
  --output-file models/smoke/quantum-smoke-f16.gguf \
  --llama-cpp-dir /path/to/llama.cpp
```

See [`docs/gguf-export.md`](docs/gguf-export.md) for verification and Windows
syntax.

## Repository structure

| Path | Purpose |
|---|---|
| `configs/` | Versioned data, tokenizer, model, training, and evaluation inputs |
| `scripts/` | Pipeline and repository-health commands |
| `tests/` | Unit, integration, slow, GPU, and network-classified tests |
| `data/evals/` | Small tracked completion prompts only |
| `reports/` | Small, reviewable preflight and smoke evidence |
| `docs/` | Architecture, operations, lineage, licensing, and release records |
| `model_cards/` | Canonical evidence-bounded model cards |

Generated `data/`, `models/`, `tokenizer/`, `exports/`, and `logs/` content is
local and ignored.

## Reproducibility

A reproducible run requires the code commit, exact configuration, dependency
environment, source and model revisions, seeds, tokenizer checksums, data
manifest, hardware/runtime record, checkpoints, evaluation outputs, and export
checksums. The repository currently has gaps for the released pilots. See
[`docs/reproducibility.md`](docs/reproducibility.md).

## Limitations and security

These models and pipelines can produce incorrect, biased, unsafe, or private
content. Dataset filtering is incomplete; pilot contexts are short; Echelon is
not trained; release licenses are incomplete; and CPU performance has not been
published reproducibly. Do not use the outputs for high-stakes decisions. Read
[`docs/limitations.md`](docs/limitations.md),
[`docs/security-model.md`](docs/security-model.md), and [`SECURITY.md`](SECURITY.md).

## Roadmap and contributing

The near-term work is license approval, stable CI, release-manifest publication,
raw evaluation evidence, reproducible CPU measurement, and Echelon data
preparation. See [`ROADMAP.md`](ROADMAP.md) and
[`CONTRIBUTING.md`](CONTRIBUTING.md). This is a sole-maintainer project and no
response time is guaranteed.

## Citation

Use [`CITATION.cff`](CITATION.cff) and include the exact code commit. Cite models,
tokenizers, and datasets separately with their own revisions, terms, manifests,
and checksums. No DOI is published.

## Licensing boundaries

Original repository source and documentation are prepared under Apache-2.0,
subject to the maintainer ownership checklist in
[`docs/licensing.md`](docs/licensing.md). That license does not cover model
weights, tokenizers, datasets, external tools, generated artifacts, or
trademarks. See [`MODEL_LICENSES.md`](MODEL_LICENSES.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Project links

- Website: [rappidAI research](https://www.rappidai-research.com)
- Hugging Face: [rappidAI](https://huggingface.co/rappidAI)
- Pilot model: [quantum-1-pilot](https://huggingface.co/rappidAI/quantum-1-pilot)
- Continued pilot: [quantum-1.6-pilot](https://huggingface.co/rappidAI/quantum-1.6-pilot)
