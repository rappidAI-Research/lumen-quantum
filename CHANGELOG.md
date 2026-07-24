# Changelog

All notable changes will be documented here. The format follows Keep a
Changelog and the project intends to use Semantic Versioning after its first
approved release.

## Unreleased

### Post-merge status

- The open-source readiness pull request (#1) is merged into `main`
  (`7e9cb34`). Dependabot then widened dependency ranges on `main` (`da79ca8`):
  `datasets<6`, `mypy<3`, `pytest<10`, `ruff<0.17`, and `setuptools<84`. The
  default CPU-only checks pass under the newest permitted `ruff`, `mypy`, and
  `pytest`.


### Added

- Open-source governance, security, contribution, support, citation, release,
  licensing, architecture, reproducibility, model-card, and application docs.
- Bounded Python metadata, Ruff, MyPy, pre-commit, CPU-only CI, repository
  health, configuration, link, and high-confidence secret checks.
- Canonical repository URL check (`scripts/check_canonical_urls.py`).
- Reproducibility run-manifest schema and builder with completion/error and
  timing fields (`schemas/run-manifest.schema.json`,
  `scripts/build_run_manifest.py`) plus a historical provenance registry and its
  validator.
- Environment capture tool (`scripts/capture_environment.py`).
- Local CPU inference benchmark harness with a dependency-free dry run
  (`scripts/benchmark_cpu_inference.py`) and prompt fixture.
- Read-only release preflight (`scripts/release_preflight.py`) and a model-card
  consistency check (`scripts/check_model_cards.py`).
- Runnable examples (`examples/`), a first-contribution guide, issue drafts,
  Hugging Face handoff drafts, a maintainer source-license approval checklist,
  and an application metrics-capture template.
- Expanded strict mypy coverage and additional CPU-only CI checks.

### Changed

- Reframed the project as rappidAI Quantum while preserving Lumen history.
- Replaced the broken llama.cpp Gitlink with an explicit external dependency.
- Canonicalized remaining repository URLs to `rappidAI-Research/lumen-quantum`.
- Expanded MODEL_LICENSES.md into a full per-artifact license matrix.

No historical release or date is inferred.
