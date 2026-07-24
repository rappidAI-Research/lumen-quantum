# Hugging Face publication checklist

This repository prepares canonical model-card source files. It does not imply
that Hugging Face repositories were changed. Apply these steps manually with an
authorized rappidAI account.

Targets:

- <https://huggingface.co/rappidAI/quantum-1-pilot>
- <https://huggingface.co/rappidAI/quantum-1.6-pilot>

## Before editing either repository

- [ ] Confirm you are signed in to the `rappidAI` organization and have write
  permission for the exact target repository.
- [ ] Open the existing Files and versions view and record the current commit.
- [ ] Confirm the public artifact files and immutable reviewed revisions still
  match the corresponding canonical card:
  - `quantum-1-pilot`: `7daf415ef09fc131d7440af8514a93fd8cf3f2a1`
  - `quantum-1.6-pilot`: `507662c095b5ba6e14f24d3fc7f0a5e29d76b7f3`
- [ ] Compare file checksums against the card. Stop if a current artifact does
  not match; investigate rather than changing the recorded revision.
- [ ] Select explicit model-weight and tokenizer reuse terms. Do not use
  `open_weights`, `license`, or similar metadata until the rights decision is
  approved and published.
- [ ] Confirm personal, confidential, or credential material is absent.

## Update `quantum-1-pilot`

- [ ] Copy the reviewed content from
  [quantum-1-pilot.md](../model_cards/quantum-1-pilot.md) into the Hugging Face
  repository `README.md`.
- [ ] Add valid Hugging Face model-card YAML only for verified facts. Keep the
  repository/model ID exact: `rappidAI/quantum-1-pilot`.
- [ ] State that reported evaluation results are publisher-reported unless the
  raw outputs and procedure are attached.
- [ ] State license status explicitly; do not infer the source-code license for
  weights or tokenizer artifacts.
- [ ] Preview the rendered card and verify headings, tables, links, and code.
- [ ] Commit with a descriptive message and record the resulting immutable
  commit in the release evidence.

## Update `quantum-1.6-pilot`

- [ ] Copy the reviewed content from
  [quantum-1.6-pilot.md](../model_cards/quantum-1.6-pilot.md) into the Hugging
  Face repository `README.md`.
- [ ] Add valid Hugging Face model-card YAML only for verified facts. Keep the
  repository/model ID exact: `rappidAI/quantum-1.6-pilot`.
- [ ] Preserve the distinction between the expected tokenizer checksum and the
  known incompatible older tokenizer checksum.
- [ ] Attach or link the final run manifest and raw evaluation outputs when
  available; otherwise label them incomplete.
- [ ] State license status explicitly and avoid an open-weight claim.
- [ ] Preview, commit, and record the new immutable card revision.

## Verification after publication

- [ ] Open both public URLs in a signed-out browser.
- [ ] Confirm the organization slug is exactly `rappidAI` in links and metadata.
- [ ] Confirm files can be resolved at the immutable revisions recorded above.
- [ ] Confirm downloads, usage examples, checksums, and artifact names point to
  the intended repository and revision.
- [ ] Confirm no unsupported hardware, benchmark, safety, or adoption claim was
  introduced.
- [ ] Update this source repository only with the new card commit IDs and any
  verified license facts, through a reviewed pull request.

## Rollback

If a published card is incorrect, revert the card-only commit on Hugging Face;
do not replace or mutate reviewed model artifacts. Record the reverted and
replacement commit IDs in the release notes or issue that tracked the change.
