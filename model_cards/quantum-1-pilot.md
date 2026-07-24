# Model card: quantum-1-pilot

## Status

Experimental legacy pilot. A public F16 GGUF exists, but explicit model-weight
and tokenizer reuse terms have not been verified as published. Do not describe
this release as open weight.

## Verified released facts

| Field | Value |
|---|---|
| Publisher | rappidAI |
| Repository | `rappidAI/quantum-1-pilot` |
| Reviewed model revision | `7daf415ef09fc131d7440af8514a93fd8cf3f2a1` |
| Historical manifest model ID | `quantum-1-base` |
| Version | `1.0.0` |
| Architecture | Llama-style causal decoder; hidden 512; intermediate 1,536; 12 layers; 8 attention heads; 8 KV heads; tied embeddings |
| Parameters | 49,295,872 |
| Vocabulary | 16,384 |
| Context | 512 tokens |
| Objective/mode | Causal completion |
| Artifact | `quantum-1-base-v1.0.0-f16.gguf` |
| Format | GGUF F16 |
| Size | 98,990,560 bytes |
| SHA-256 | `aeab97e50a5789772b69cf1554ba74eb915b5621835d80d40785b473b62fd1a5` |
| Reviewed code revision | `f7eda1fb0ae153f0f9cc3477ead997cbdb462b39` |

## Publisher-reported training information

The release is described as an independently pretrained German completion
pilot, with approximately 100M training tokens from the FineWeb2-HQ-based pilot
pipeline. A final versioned run manifest, complete logs, resolved environment,
hardware/runtime record, and raw evaluation bundle are not linked here.

## Tokenizer

A custom SentencePiece tokenizer with 16,384 entries is reported. The exact
public tokenizer artifact revision, checksum, training manifest, and license
must be confirmed before independent reuse.

## Intended use

- historical research reference for a compact from-scratch pipeline;
- local completion experiments in a controlled environment; and
- tests of model/tokenizer/GGUF tooling.

## Prohibited or high-risk use

Do not use for medical, legal, financial, safety-critical, consequential,
surveillance, or factual-reliability tasks. Do not present outputs as expert
advice or as aligned chat responses. Review applicable law and artifact terms.

## Limitations

The model is very small, German-focused, completion-only, limited to 512 tokens,
not instruction tuned, not safety aligned, and expected to produce incoherent,
biased, private, harmful, or false output. No standardized downstream benchmark
or reproducible CPU inference measurement is published.

## Evaluation provenance

No raw versioned standardized evaluation output is available in this repository.
Any qualitative observations should be treated as exploratory.

## Hardware

Training and inference hardware/runtime are unknown unless supplied by a
separate versioned record. No minimum RAM, latency, energy, or throughput claim
is made.

## License

Model-weight and tokenizer terms are unresolved. Apache-2.0 for this code
repository does not apply to the GGUF or tokenizer.

## Citation and sources

Cite the model repository and reviewed revision, artifact manifest/checksum, and
code revision separately:

- https://huggingface.co/rappidAI/quantum-1-pilot
- https://github.com/rappidAI-Research/lumen-quantum/tree/f7eda1fb0ae153f0f9cc3477ead997cbdb462b39
