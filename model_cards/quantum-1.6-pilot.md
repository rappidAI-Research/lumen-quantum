# Model card: quantum-1.6-pilot

## Status

Experimental continued-pretraining pilot. A public F16 GGUF exists, but explicit
model-weight and tokenizer reuse terms have not been verified as published. Do
not describe this release as open weight.

## Verified released facts

| Field | Value |
|---|---|
| Publisher | rappidAI |
| Repository | `rappidAI/quantum-1.6-pilot` |
| Reviewed model revision | `507662c095b5ba6e14f24d3fc7f0a5e29d76b7f3` |
| Version | `1.6.0` |
| Architecture | Llama-style causal decoder; hidden 512; intermediate 1,536; 12 layers; 8 attention heads; 8 KV heads; tied embeddings |
| Parameters | 49,295,872 |
| Vocabulary | 16,384 |
| Context | 512 tokens |
| Objective/mode | Continued causal pretraining; completion |
| Artifact | `quantum-1.6-pilot-v1.6.0-f16.gguf` |
| Format | GGUF F16 |
| Size | 98,990,560 bytes |
| SHA-256 | `6bda15fcd51286e55174d5876fe44aa9518fb18b75fb5aa4f7402ebd039bd994` |
| Reviewed code revision | `f7eda1fb0ae153f0f9cc3477ead997cbdb462b39` |

## Publisher-reported training and evaluation

- approximately 100M earlier pilot tokens plus 500M additional German tokens;
- validation loss 3.348852; and
- perplexity 28.4700.

These are publisher-reported. This repository does not contain the final run
manifest, complete logs, raw evaluation output, resolved environment, or
hardware/runtime evidence needed to independently reproduce them. The 30,518
steps in YAML are a configured target, not a verified completed count.

## Tokenizer

The code requires the frozen 16,384-entry tokenizer with SHA-256
`be99b72377f3cb2ce1c875103d0324a2001ee5543a49e7c8fabfc1e384b1b6f6` and rejects
the incompatible older pilot tokenizer. Public artifact revision and license
still require confirmation.

## Intended use

- study of documented weights-only continued pretraining;
- controlled German completion experiments; and
- local GGUF/tooling compatibility investigation.

## Prohibited or high-risk use

Do not use for medical, legal, financial, safety-critical, consequential,
surveillance, or factual-reliability tasks. Do not present it as a chat assistant
or an aligned production model.

## Limitations

The model is very small, short-context, completion-only, not instruction tuned,
not safety aligned, and can generate incoherent, biased, private, harmful, or
false text. No downstream benchmark suite, uncertainty analysis, reproducible
CPU measurement, or broad client-compatibility report is published.

## Hardware

Training and inference hardware/runtime are unknown unless supplied in a
separate versioned record. No minimum RAM, latency, throughput, energy, or cost
claim is made.

## License

Model-weight and tokenizer terms are unresolved. Apache-2.0 for this code
repository does not apply to the GGUF or tokenizer.

## Citation and sources

- https://huggingface.co/rappidAI/quantum-1.6-pilot
- https://github.com/jonascikemgil07-hue/lumen-quantum/tree/f7eda1fb0ae153f0f9cc3477ead997cbdb462b39

Cite the exact model revision, artifact checksum, code revision, and dataset
terms rather than only the display name.
