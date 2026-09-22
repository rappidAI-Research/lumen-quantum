# Quantum 1 Echelon 1B — Garden v2 source review

Status: public source metadata and immutable candidate revisions are captured,
but **no Garden v2 source is production-approved yet**. This file records
engineering evidence; it is not legal advice and does not silently accept
dataset terms on behalf of the maintainer.

## Gate

A source may enter the production corpus only after all of the following are
explicitly recorded in `configs/echelon/1b/sources.yaml`:

1. exact dataset/config selection and immutable revision;
2. provenance and public license/terms evidence;
3. project-level terms/rights review;
4. a removal/update process compatible with the source obligations;
5. `production_approved: true` set by an explicit maintainer decision.

`python scripts/validate_echelon_1b.py --require-production-sources` is
expected to fail until those decisions are complete. Normal planning validation
remains green so engineering can continue without pretending the legal gate is
closed.

## Captured candidates

| Role | Candidate / revision | Public evidence captured | Current blocker |
|---|---|---|---|
| German HQ web | `epfml/FineWeb2-HQ` / `c0c06e94fd3a44ae9e802b2b0fc533817601eb5e` / `deu_Latn` | Dataset metadata reports ODC-By 1.0 plus Common Crawl Terms; dataset card warns about PII/sensitive content and provides upstream opt-out/removal guidance. | Human terms/rights approval plus project removal workflow. |
| English education | `HuggingFaceFW/fineweb-edu` / `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` / `default` | The pinned v1.4.0 state reports ODC-By 1.0 and Common Crawl Terms. | Human terms/rights approval plus project removal workflow. |
| Reference knowledge | Not selected | None. | Source selection, immutable revision, provenance and rights review. |
| Mathematics | `HuggingFaceTB/finemath` / `e92b25a616738fe95dc186b64dfb19f9c8525594` / `finemath-4plus` | Dataset card reports FineMath-4+ as the 9.6B-token high-quality subset and ODC-By 1.0 plus Common Crawl Terms. | Human terms/rights approval plus project removal workflow. |
| Code | `HuggingFaceTB/stack-edu` / `eeec5caac5cc3758a18f1d3ba4416837a9ba814c` | Stack-Edu contains Software Heritage IDs rather than source text and defers content licensing to The Stack v2. The Stack v2 terms require a Software Heritage/INRIA agreement for bulk content access, adherence to original repository licenses, and keeping data current with validated removals. | Explicit terms/access decision, license-handling policy, removal/update process, and exact language/config mix. |

## Engineering consequences

- The shared tokenizer A/B corpus assembler must reject unapproved sources in
  production mode.
- No 40B Garden v2 production manifest may be declared ready while this source
  gate fails.
- Public availability is never treated as permission.
- Revisions above freeze candidate evidence only; they do not force the final
  production source mix.
- Stack-Edu remains especially constrained because identifiers and source
  content have different access/licensing layers.
- The reference/knowledge slot is deliberately unresolved rather than silently
  filled with an arbitrary encyclopedia source.

## Evidence URLs

- https://huggingface.co/datasets/epfml/FineWeb2-HQ
- https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- https://huggingface.co/datasets/HuggingFaceTB/finemath
- https://huggingface.co/datasets/HuggingFaceTB/stack-edu
- https://huggingface.co/datasets/bigcode/the-stack-v2
- https://commoncrawl.org/terms-of-use/
