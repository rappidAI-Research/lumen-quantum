# Roadmap

This roadmap is directional, not a delivery promise.


## Priority: Quantum 1 Echelon 1B

- Treat [`docs/echelon/1b/STATUS.md`](docs/echelon/1b/STATUS.md) as the short
  operational state and the 1B execution contract as the current strategic path.
- Freeze exactly one 32K/48K tokenizer+architecture candidate from retained A/B
  evidence; the historical 506M preflight remains untouched.
- Complete Garden v2 source review, decontamination, 40B sharding, checksums and
  final manifest before paid Base training.
- Implement a production sharded loader with exact data-position resume and
  prove checkpoint recovery plus client-disconnect-safe unattended execution.
- Keep SFT, preference training and Chat evaluation as mandatory deliverables;
  the Base checkpoint is not the project endpoint.
- Do not start a large H100 session until Base and Chat smoke paths, quota,
  current pricing, budget controls and recovery gates are all green.

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
