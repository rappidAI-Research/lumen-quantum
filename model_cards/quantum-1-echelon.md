# Model card: quantum-1-echelon

## Status

In development. **Quantum 1 Echelon** is the current strategic model line. The
repository is now preparing a new approximately 1B-parameter Base stage followed
by a required Chat stage. No 1B production dataset, trained checkpoint, GGUF,
capability evaluation or model-weight license exists yet.

The earlier 506M Echelon architecture/tokenizer/Garden work is retained as
historical preflight evidence. It is not presented as the final production
architecture and is not overwritten by the 1B work.

## Current 1B target (planning, not execution evidence)

| Field | Current target / state |
|---|---|
| Model line | `quantum-1-echelon` |
| Public name | Quantum 1 Echelon |
| End product | Chat stage; Base is an intermediate stage |
| Base parameter target | approximately 1.0–1.02B |
| Context target | 4,096 tokens |
| Precision target | BF16 |
| Attention | grouped-query attention (GQA) |
| Candidate A | 32,768 vocab / 21 layers / ~1.014B parameters |
| Candidate B | 48,000 vocab / 20 layers / ~1.000B parameters |
| Base token target | 40B high-quality tokens |
| Conditional extension | 50B only after the 40B quality/budget gate |
| Chat post-training | SFT then preference training/evaluation |
| Production checkpoint | Not available |

Candidate configurations live under [`configs/echelon/1b/`](../configs/echelon/1b/).
Exactly one architecture/tokenizer candidate may be frozen after the tokenizer
A/B evidence exists.

## Historical 506M preflight evidence

The retained historical preflight is useful evidence that the architecture,
tokenizer and Garden tooling existed before the 1B redesign. It is not a trained
model.

| Field | Retained historical value |
|---|---|
| Architecture | Llama-style causal decoder |
| Parameters | 506,333,440 |
| Vocabulary target | 32,768 |
| Context target | 2,048 tokens |
| Hidden/intermediate | 1,280 / 3,584 |
| Layers | 26 |
| Attention/KV heads | 20 / 5 |
| Embeddings | Tied |
| Reviewed code revision | `f7eda1fb0ae153f0f9cc3477ead997cbdb462b39` |

The corresponding report remains at
[`reports/quantum-1-echelon/quantum-1-echelon-base-preflight.json`](../reports/quantum-1-echelon/quantum-1-echelon-base-preflight.json).

## Tokenizer status

The historical Echelon tokenizer pipeline used SentencePiece BPE with byte
fallback and a 32,768 vocabulary; the tracked report recorded 23 round-trip
cases with zero failures. Those reports remain historical evidence.

For the 1B line, 32K and 48K tokenizer candidates are now explicit planning
artifacts. Neither candidate is frozen or claimed trained. The A/B gate must
compare German, English, Code and Math token efficiency, byte fallback,
round-trip behavior and checksums before the production architecture is frozen.

## Data and objective

The objective remains causal language modeling. Garden v2 plans a German-first,
multi-source 40B-token Base corpus with stable held-out splits, PII/sensitive
filtering, exact deduplication, source-specific near-duplicate policy,
benchmark decontamination, 4K packing, 100M-token shards, checksums and a final
manifest.

The currently committed source registry contains candidates, not blanket
approval. Immutable revisions, terms/rights review, PII/removal handling and
source-specific provenance must be completed before production ingestion.

The earlier Garden smoke saw 5,001 documents, accepted 1,559 and produced
1,380,886 tokens. Those numbers validate the old smoke pipeline only; they are
not evidence for the 1B production corpus.

## Chat-stage requirement

The project does not stop at a Base checkpoint. After Base evaluation, the plan
requires supervised instruction tuning, a preference-training comparison and a
separate Chat evaluation. If preference training regresses quality, the better
SFT checkpoint remains the Chat candidate rather than being overwritten by a
weaker result.

## Incomplete work

- freeze the 1B architecture/tokenizer candidate;
- complete source rights/terms/removal review;
- implement and execute Garden v2 production preparation;
- produce the final data manifest, shard checksums and decontamination report;
- prove the production shard loader and exact data-position resume;
- prove unattended/client-disconnect-safe execution and external recovery;
- run LR calibration and Base training;
- retain raw Base evaluation evidence;
- prepare and review SFT/preference data;
- train and evaluate the Chat stage;
- select model/tokenizer release terms; and
- export and verify release artifacts.

See [`docs/echelon/1b/STATUS.md`](../docs/echelon/1b/STATUS.md) for the short
operational state and [`docs/echelon/1b/README.md`](../docs/echelon/1b/README.md)
for the repository-side execution contract.

## Intended use

Current committed 1B artifacts are intended for architecture, tokenizer, data,
runtime and post-training preparation. They are not usable as model inference
artifacts because no 1B trained weights exist.

## Prohibited or high-risk use

Do not claim a released or trained 1B Echelon model, use nonexistent Echelon
weights for high-stakes tasks, or publish weights/tokenizers/data without the
required provenance, security and license approvals.

## Limitations

Architecture and pipeline plans do not predict final quality, safety, latency,
memory use, data suitability or training success. The project remains
compute-constrained relative to industrial 1B models trained on much larger
corpora. Every quality claim must come from retained raw evaluation evidence.

## License and citation

No 1B model-weight or tokenizer license is selected. Source code is separately
licensed under Apache-2.0. Cite the exact code revision and retained reports; do
not cite the planning target as a model release.
