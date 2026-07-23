## Purpose

Describe the problem, the smallest useful change, and its evidence.

## Change type

- [ ] Code or configuration
- [ ] Data/tokenizer/model pipeline
- [ ] Tests or CI
- [ ] Documentation or model card
- [ ] Security, licensing, or maintenance

## Validation

- [ ] `ruff format --check .`
- [ ] `ruff check .`
- [ ] `mypy`
- [ ] fast CPU tests (`not slow and not gpu and not network`)
- [ ] configuration, repository-health, link, and secret checks
- [ ] additional scoped checks documented below

## Reproducibility and provenance

List exact data, model, tokenizer, tool, and code revisions. Separate measured
results, publisher-reported results, configuration targets, and incomplete work.

## Licensing and security

Identify new third-party material and its terms. Confirm that no credentials,
personal data, restricted data, large artifacts, or untrusted checkpoints are
included. Describe any security impact and follow `SECURITY.md` for private issues.

## Resource use

State whether this requires network, slow, GPU, or large-download validation.
Do not trigger expensive workloads as part of review without maintainer approval.
