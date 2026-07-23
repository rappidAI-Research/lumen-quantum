# Release process

## Versioning

Code releases use SemVer-style tags. Python metadata uses the PEP 440 equivalent
(`0.1.0a0` for proposed tag `v0.1.0-alpha`). Model, tokenizer, dataset, and GGUF
artifacts are separate releases with separate terms and manifests.

## Candidate workflow

1. Freeze scope to functionality verified by tests.
2. Confirm source ownership and license approval.
3. Resolve or explicitly exclude model/tokenizer/data rights.
4. Run the full release checklist on a clean checkout.
5. Review changelog, README, model cards, security, and compatibility.
6. Produce release notes without adoption or capability inflation.
7. Obtain maintainer approval.
8. Create the signed/annotated tag and GitHub release manually.

No tag, PyPI upload, model publication, or merge is performed by this readiness
change.
