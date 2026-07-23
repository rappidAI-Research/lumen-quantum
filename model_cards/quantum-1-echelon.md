# Model card: quantum-1-echelon

## Status

In development. This is a model-line and pipeline card, not a released trained
model card. No checkpoint, GGUF, model revision, capability evaluation, or
model-weight license exists.

## Verified preflight facts

| Field | Value |
|---|---|
| Model line | `quantum-1-echelon` |
| Stage represented | Echelon Base architecture preflight |
| Architecture | Llama-style causal decoder |
| Parameters | 506,333,440 |
| Vocabulary target | 32,768 |
| Context target | 2,048 tokens |
| Hidden/intermediate | 1,280 / 3,584 |
| Layers | 26 |
| Attention/KV heads | 20 / 5 |
| Embeddings | Tied |
| Reviewed code revision | `f7eda1fb0ae153f0f9cc3477ead997cbdb462b39` |
| Model revision/checksum | Not available; no model artifact exists |

## Tokenizer preflight

The configured tokenizer is SentencePiece BPE with byte fallback and a 32,768
vocabulary. The tracked report records 23 round-trip cases with zero failures.
The checksum record lists:

- model: `97dce887ba56afc3f99a6de17f88621dfa527f48a9f455801064cceb0d4d1dd6`;
- vocabulary: `0fdfef7bb9c5f07760004fbc4ecda6ade2e99efa9cd71d74df3e63a004be82c5`;
- tokenizer config: `36d3a745a64c1901c5f03c6810d94719b705dac1ceddd7de21af1f79d9fa9b80`.

The binaries are not tracked and no tokenizer release license is selected.

## Data and objective

The planned objective is causal language modeling. The Garden configuration
references FineWeb2-HQ `deu_Latn` at revision
`c0c06e94fd3a44ae9e802b2b0fc533817601eb5e`. A smoke run saw 5,001 documents,
accepted 1,559, and produced 1,380,886 tokens. These numbers validate the
pipeline only. The production target is configured but the production run has
not started.

## Incomplete work

- production-data preparation;
- final data manifest, removal handling, and checksums;
- model training and checkpoints;
- raw evaluation and capability testing;
- runtime/hardware measurements;
- Base-to-Chat adaptation design and upstream license review; and
- model/tokenizer release terms.

## Intended use

Current artifacts are intended for architecture, tokenizer, configuration, and
data-pipeline review. They are not usable for model inference.

## Prohibited or high-risk use

Do not claim a released or trained Echelon model, use it for high-stakes tasks,
or publish weights/tokenizers/data without the required provenance, security,
and license approvals.

## Limitations

Architecture and pipeline preflights do not predict quality, safety, latency,
memory, data suitability, or training success. The web-derived data risks in
`docs/datasets.md` remain.

## License and citation

No model-weight or tokenizer license is selected. Source code is separately
prepared under Apache-2.0. Cite the exact code revision and tracked reports in
`reports/quantum-1-echelon/`; do not cite this as a model release.
