# Release checklist

Target: `v0.1.0-alpha`
State: draft preparation only

## Ownership and licensing gate

- [ ] Jonas Désiré Cikemgil confirms he owns or is authorized to license all
  source code and documentation covered by the top-level Apache-2.0 notice.
- [ ] No employer, client, collaborator, copied snippet, or prior agreement
  imposes incompatible terms.
- [ ] `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, `DATA_SOURCES.md`,
  `MODEL_LICENSES.md`, and `docs/licensing.md` are reviewed and accurate.
- [ ] No model weight, tokenizer, dataset, third-party tool, or brand material
  is accidentally represented as Apache-2.0-covered.

## Technical gate

- [ ] The release commit is the reviewed pull-request head on `main`.
- [ ] Python 3.11 and 3.12 CI jobs pass.
- [ ] Ruff formatting and lint pass.
- [ ] Configured type checks pass.
- [ ] Fast CPU tests pass with slow, GPU, and network tests excluded.
- [ ] YAML, JSON, JSONL, and CFF validation passes.
- [ ] Repository-health, Markdown-link, and secret-pattern checks pass.
- [ ] No tracked Git link lacks matching submodule metadata.
- [ ] No large model, tokenizer, dataset, checkpoint, or generated artifact is
  included in Git or the source archive.

## Documentation gate

- [ ] README commands match the release commit and use relative/cross-platform
  paths.
- [ ] Model-family names and status are consistent.
- [ ] Release notes contain only verified functionality.
- [ ] Limitations, security risks, provenance, and unresolved licensing are
  prominent.
- [ ] Canonical model cards match the immutable public artifact revisions and
  checksums they cite.
- [ ] Changelog has a dated `0.1.0-alpha` entry prepared from `Unreleased`.
- [ ] Citation metadata validates and matches the release identifier.

## Publication gate

- [ ] Maintainer explicitly approves the alpha release.
- [ ] Create annotated tag `v0.1.0-alpha` from the reviewed merge commit.
- [ ] Create a GitHub prerelease using
  [v0.1.0-alpha.md](v0.1.0-alpha.md), attaching no generated ML artifact by
  default.
- [ ] Verify the source archive excludes ignored runtime material.
- [ ] Verify release links in a signed-out browser.
- [ ] Record tag, commit, CI run, and release URLs in the release evidence.

Do not tag, publish, or upload artifacts while any required item is unchecked.
