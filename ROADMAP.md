# Roadmap

This roadmap is directional, not a delivery promise.

## Before v0.1.0-alpha

- Retain the recorded [source-license approval](docs/maintainer-source-license-approval.md) and complete the remaining release-checklist items.
- Keep default CI green on Python 3.11 and 3.12.
- Verify the external llama.cpp revision and CPU GGUF wrapper smoke test.
- Publish explicit pilot artifact notices consistent with the maintainer's recorded all-rights-reserved decision; no reuse grant is inferred.

## Research readiness

- Publish final pilot run manifests and raw evaluation outputs.
- Add reproducible CPU inference measurements with hardware and runtime details.
- Follow the [Echelon compute plan](docs/compute-plan.md): review data rights and
  budget, complete production-data preparation, then validate a full-context
  training recipe before separately authorizing training and evaluation.
- Keep production execution and missing evidence tracked in issues #2–#6;
  documentation does not satisfy those run and publication gates.
- Define an openly licensed adaptation baseline with exact upstream provenance.

## Maintenance

- Reduce duplicated historical commands and config drift.
- Add release automation only after the manual alpha process is proven.
- Expand governance only when real additional maintainers participate.
