# Repository audit

Audit date: 2026-07-22
Repository: `jonascikemgil07-hue/lumen-quantum`
Audited base revision: `f7eda1fb0ae153f0f9cc3477ead997cbdb462b39`
Readiness branch: `chore/core-oss-readiness`

This is the technical repository for rappidAI Quantum. The separate
`rappidai-research.com` repository is the project website and research index;
it is not the model-development core.

## Executive result

The repository contains substantive data, tokenizer, model, training,
checkpoint/resume, evaluation, and GGUF tooling. The readiness branch turns it
into a conventional Python project, adds CPU-only CI and repository health
checks, removes a broken `llama.cpp` Git link, documents provenance boundaries,
and prepares—but does not publish—an Apache-2.0 source release.

The codebase is suitable for a draft `v0.1.0-alpha` release after CI passes and
the maintainer completes the approval items below. It is not evidence of a
production-ready model, completed Echelon training, benchmark leadership, broad
adoption, or open-weight licensing.

## Actual project scope and architecture

- `scripts/`: 39 tracked Python entry points on the audited base revision.
- `configs/`: 18 tracked YAML configurations for pilot, final-plan, diagnostic,
  and Echelon workflows.
- `tests/`: 18 tracked test modules on the audited base revision; this branch
  adds repository-health tests and centralized marker classification.
- `docs/`: 10 tracked documents on the audited base revision; this branch adds
  maintained English-first guides while preserving historical procedures.
- `reports/`: 7 tracked, experiment-specific reports.
- `evals/`: one small tracked JSONL evaluation fixture/result file.
- Ignored runtime areas include datasets, tokenizers, checkpoints, models,
  logs, and generated exports. They are not source release artifacts.

The operational flow is:

1. source and filter text data;
2. train or validate a tokenizer;
3. validate model configuration and size;
4. run smoke or training workflows with resumable checkpoints;
5. evaluate saved artifacts;
6. export compatible artifacts through an explicitly supplied external
   `llama.cpp` checkout.

See [architecture.md](architecture.md) and [model-lineage.md](model-lineage.md).

## Working and safe commands

The following commands are designed to work without a GPU or network access
after the development environment is installed:

```bash
python -m scripts.make_smoke_data --output data/smoke/train.txt
python -m scripts.repository_health
python -m scripts.validate_repository_configs
python -m scripts.check_markdown_links
python -m scripts.check_secrets
ruff format --check .
ruff check .
mypy
pytest -m "not slow and not gpu and not network"
```

Commands that prepare datasets, install ML dependencies, train models, load
external artifacts, or use a cloud/GPU environment are intentionally outside
the default health check. Their prerequisites and side effects are documented
in the topic guides.

## Broken or unsafe commands found

| Finding on the base revision | Impact | Readiness-branch treatment |
| --- | --- | --- |
| `tools/llama.cpp` was a Git link without `.gitmodules` | A normal clone could not initialize the dependency | Git link removed; callers require `--llama-cpp-dir`; tested upstream revision recorded |
| GGUF/diagnostic scripts assumed a repository-local or `/srv` checkout | Non-portable and could silently target the wrong tool | External path must be explicit |
| Historical docs used `C:\LumenQuantum` | Personal/project-specific commands | Replaced with neutral example paths and historical banners |
| Several dataset configs use mutable `revision: main` | Exact historical data cannot be reconstructed | Not guessed or silently changed; actual run revisions remain a maintainer follow-up |
| Resume logic loads trusted optimizer state with `weights_only=False` | A malicious checkpoint can execute pickle payloads | Security boundary documented; only locally produced, trusted resume state is supported |

## Test inventory

Tests cover configuration invariants, data cleaning/splitting/tokenization,
tokenizer rules, parameter counts and model shape, checkpoint saving/resume,
evaluation helpers, GGUF command construction, generation diagnosis, pilot
preflight behavior, and Echelon configuration/preflight/tokenizer cases.

This branch classifies tests as `unit`, `integration`, `slow`, `gpu`, or
`network`, adds strict marker handling, and adds six repository-level tests for
Git-link metadata, tracked secrets, large files, config parsing, Markdown
links, and high-confidence secret patterns. Default CI excludes `slow`, `gpu`,
and `network` tests and uses CPU wheels only.

Some ML-facing tests require the optional `ml` dependencies. They must run in
the CPU CI job; the local lightweight validation path does not download PyTorch,
models, or datasets.

## Licensing status

The base revision had no top-level license. This branch prepares Apache License
2.0 for original source code and repository documentation, plus `NOTICE`,
third-party notices, and explicit licensing boundaries. That preparation is not
a factual ownership determination.

Before merge, Jonas Désiré Cikemgil must confirm that he has authority to
license every covered contribution and that no employer, client, collaborator,
or copied source imposes conflicting terms. Model weights, tokenizer artifacts,
datasets, external tools, and trademarks are not granted Apache-2.0 merely by
being referenced here. See [licensing.md](licensing.md),
[../DATA_SOURCES.md](../DATA_SOURCES.md), and
[../MODEL_LICENSES.md](../MODEL_LICENSES.md).

## Provenance risks

- Git history has one public maintainer identity but includes inconsistent
  historical author formatting. This is not evidence of third-party ownership.
- No copied code block with an identified incompatible license was found during
  the text audit. Maintainer knowledge is still required to confirm origin.
- The removed `llama.cpp` link pointed to upstream commit
  `d4cff114c0084f1fbc9b4c62717eca8fb2ae494a`, which is MIT-licensed upstream;
  no `llama.cpp` source is distributed by this branch.
- FineWeb2-HQ data is governed by ODC-By 1.0 and Common Crawl terms. Web-derived
  content may retain third-party rights and personal information.
- Pilot model and tokenizer repositories are public, but explicit artifact
  reuse terms have not been verified as published. They are not described as
  open weight.
- Reports are experiment-specific. Measurements without retained raw outputs,
  environment metadata, and immutable revisions are publisher-reported, not
  independently verified benchmarks.

## Reproducibility gaps

- Historical pilot configs that name dataset revision `main` need their actual
  immutable run revision recorded.
- The final `quantum-1.6-pilot` run manifest and complete raw evaluation outputs
  are not published in this repository.
- Hardware and timing records are incomplete for some historical runs.
- A repeatable CPU inference measurement with environment capture is absent.
- The Echelon production-data run and training are incomplete.
- Ignored model/tokenizer artifacts cannot be reconstructed from Git alone.

## Documentation and naming findings

The base README was a long procedural German document and mixed the earlier
Lumen identity with several quantum stages. The maintained README is now
English-first and moves details into topic guides. Historical documents retain
their language and technical value but are marked as historical.

Canonical naming is:

- project: rappidAI Quantum;
- historical internal/project name: Lumen;
- independent pilot stages: `quantum-1-pilot` and `quantum-1.6-pilot`;
- strategic model line: `quantum-1-echelon`;
- Echelon Base and Echelon Chat: stages or variants inside that line.

No repository, file, or public model was renamed. A separate recommendation is
in [repository-rename-recommendation.md](repository-rename-recommendation.md).

## Ignored artifacts and large-file risks

The audit found no tracked model weights, tokenizer builds, large datasets, or
other individual files above 10 MiB. The working tree contained no ignored
runtime artifact directory at audit time. Generated checkpoints, `.safetensors`,
`.gguf`, `.bin`, and dataset shards must remain outside normal Git history;
releases should use purpose-built artifact storage with checksums and terms.

The repository-health check fails on tracked model-like artifacts, tracked
`.env` files, unmatched Git links, and files over the configured threshold.

## Security concerns

- Resume checkpoints may include Python-pickled optimizer state. Never load an
  untrusted resume directory.
- Model, tokenizer, dataset, and GGUF inputs are untrusted data; review sources,
  revisions, licenses, checksums, and parsers before use.
- Training and generation output can reproduce unsafe, biased, false, or private
  content. The repository offers no safety certification.
- Dataset acquisition and external `llama.cpp` execution can involve network or
  native-code risk and are excluded from default CI.
- Secret scanning is deliberately high-confidence and is not a substitute for
  GitHub secret scanning, dependency review, or credential rotation.

See [security-model.md](security-model.md) and [../SECURITY.md](../SECURITY.md).

## Release readiness

Prepared on this branch:

- version metadata `0.1.0a0` and draft `v0.1.0-alpha` notes;
- bounded Python 3.11/3.12 dependency ranges;
- CPU-only CI, developer tooling, health checks, and community files;
- model cards and provenance/licensing documentation;
- release checklist with explicit human approvals.

Not done: no tag, GitHub release, package publication, model publication,
dataset download, GPU run, or Echelon production run was created.

## Remaining maintainer decisions

- [ ] Confirm source/documentation ownership and approve Apache-2.0 before merge.
- [ ] Confirm `NOTICE` names and years are accurate.
- [ ] Choose and publish explicit pilot model-weight and tokenizer licenses.
- [ ] Verify the exact immutable dataset revision used by each historical run.
- [ ] Decide whether and when to rename `lumen-quantum` to `rappidai-quantum`.
- [ ] Approve `v0.1.0-alpha` only after required CI is green.
- [ ] Decide whether Hugging Face model-card updates should be applied manually.
- [ ] Review repository and profile wording before submitting the OpenAI
  Codex for Open Source application.

Until those items are resolved, the readiness work should remain a draft pull
request and no release should be tagged.
