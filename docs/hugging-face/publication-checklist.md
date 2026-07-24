# Hugging Face publication checklist (prepared, not executed)

These drafts are manual publication templates. Nothing here is pushed
automatically. Publishing to Hugging Face is a human decision.

## Before publishing any card

- [ ] The Hugging Face card matches the canonical GitHub card in
      [../../model_cards/](../../model_cards/).
- [ ] Every open license field is still shown as open, not silently filled.
- [ ] No artifact is described as "open weight" without a published reuse
      license (see [../../MODEL_LICENSES.md](../../MODEL_LICENSES.md)).
- [ ] Exact model revision, file name, size, and SHA-256 match the artifact.
- [ ] Intended-use, out-of-scope, and limitation sections are present.
- [ ] `quantum-1-echelon` is described as in development, with no checkpoint,
      no weights, no weight license, and no capability benchmark.

## Drafts

- [quantum-1-pilot-card-draft.md](quantum-1-pilot-card-draft.md)
- [quantum-1.6-pilot-card-draft.md](quantum-1.6-pilot-card-draft.md)
- [quantum-1-echelon-card-draft.md](quantum-1-echelon-card-draft.md)

## Human decisions still required

- [ ] Publish or explicitly withhold pilot weight licenses.
- [ ] Publish or explicitly withhold tokenizer licenses.
- [ ] Approve each card before it is pushed.
