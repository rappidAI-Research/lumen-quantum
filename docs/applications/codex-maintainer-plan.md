# Codex maintainer plan

This plan defines practical, reviewable uses of Codex for the sole maintainer of
rappidAI Quantum. Codex assists; the maintainer remains accountable for every
merge, license decision, release, credential, and external publication.

## Routine maintenance

| Workflow | Codex contribution | Human control |
| --- | --- | --- |
| Issue reproduction | Build minimal CPU-only reproductions, classify environment and scope, propose regression tests | Confirm user impact and priority |
| Pull-request review | Inspect diffs, tests, provenance, public API changes, and security boundaries | Decide requested changes and merge |
| Test expansion | Add deterministic unit tests and small fixtures; identify untested branches | Approve behavior and resource budget |
| Config validation | Parse YAML/JSON/CFF, enforce invariants, compare manifests with scripts | Approve configuration semantics |
| Dependency updates | Review release notes, bounds, compatibility, and CI results | Approve upgrades and lock changes |
| Model-card consistency | Compare architecture, revisions, checksums, licenses, and stated evidence | Verify source artifacts and publisher claims |
| Metadata synchronization | Prepare consistent GitHub, website, and Hugging Face links and status text | Authorize and execute external publication |
| Security review | Identify unsafe deserialization, secret exposure, dependency, subprocess, and input risks | Triage privately and coordinate disclosure |
| Documentation | Keep quickstarts, historical context, cross-platform commands, and limitations aligned | Approve public claims |
| Releases | Draft changelog and release notes from verified changes; run checklist | Approve tag and publish release |

## Proposed cadence

- Per pull request: formatting, lint, focused tests, provenance review, and a
  concise risk summary.
- Weekly when active: dependency and issue triage, stale-document review, and
  roadmap reconciliation.
- Before a model-card update: immutable artifact, checksum, license, evaluation,
  and link comparison.
- Before a release: clean-clone CPU verification, full CI review, documentation
  audit, source archive inspection, and explicit maintainer sign-off.

## Guardrails

- No GPU training, large model/data download, production pipeline, or expensive
  workload without an explicit scoped request and budget.
- No invented result, adoption, contributor, benchmark, or release evidence.
- No external issue, comment, model-card change, tag, release, or merge without
  task-specific authorization.
- No credential in prompts, logs, fixtures, commits, or reports.
- No loading an untrusted pickle-containing checkpoint.
- No license inference from repository location, model architecture, or source
  code license.
- Prefer immutable revisions and checksums; record unknowns rather than filling
  them with assumptions.

## First six-month outcomes

If support is granted, prioritize maintenance quality over visible activity:

1. keep CPU CI reliable across supported Python versions;
2. close reproducibility gaps with immutable run manifests and raw evidence;
3. publish explicit model/tokenizer terms after rights review;
4. add a small repeatable CPU inference measurement;
5. keep canonical model cards synchronized across repositories;
6. review and document a reproducible open-weight adaptation baseline only if
   its upstream license and provenance are suitable;
7. prepare releases from verified changes, without promising a fixed cadence.

Success is measured by reproducible maintenance evidence and reduced unresolved
risk, not by issue counts, generated commits, or unsupported adoption claims.
