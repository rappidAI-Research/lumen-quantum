# Project specification: rappidAI Quantum

## Scope

rappidAI Quantum provides reproducible pipelines for preparing data, training
tokenizers, developing, evaluating, and exporting compact language models. It
is the technical core repository. The website is a separate publication and
research-index surface.

## Identity and history

- **Current project identity:** rappidAI Quantum.
- **Historical name:** Lumen was an earlier internal/project name and remains in
  legacy paths and source identifiers.
- **Pilot stages:** `quantum-1-pilot` and `quantum-1.6-pilot` are independently
  pretrained experimental stages in the historical pilot line.
- **Strategic line:** `quantum-1-echelon` is the current strategic model line.
- **Echelon stages:** Echelon Base and Echelon Chat are variants or stages within
  that line, not separate families.

Repository, file, and public-model renames require a separate compatibility and
redirect plan.

## Design principles

1. Separate verified evidence, publisher reports, configuration targets, and
   incomplete work.
2. Pin code, dataset, upstream-model, tokenizer, and tool revisions for releases.
3. Keep data, weights, tokenizers, logs, secrets, and large artifacts out of Git.
4. Make CPU-only validation the safe default; classify network, slow, and GPU
   tests explicitly.
5. Treat model/data licensing and security as independent release gates.
6. Preserve local checkpoint/resume workflows across Windows, Linux, and macOS.

## Weight-origin rule

The historical pilot line deliberately created models from random weights and
did not load third-party pretrained weights. That remains a requirement for
reproducing those experiments.

It is not a universal restriction on future fine-tuning. Every derived model
must record:

- upstream model and exact immutable revision;
- upstream license and any acceptable-use terms;
- tokenizer identity and revision;
- changed architecture or vocabulary behavior;
- training data terms and provenance; and
- a release license compatible with all inputs.

## Technical contract

- Python: 3.11 or 3.12.
- Configurations: YAML, validated before work begins.
- Models: Llama-style causal decoders constructed from explicit config.
- Tokenizers: local SentencePiece artifacts with checksums and special-token IDs.
- Checkpoints: model, optimizer, scheduler, RNG, step, epoch, and configuration.
- Evaluation: validation loss/perplexity plus fixed completion prompts; raw output
  is required before promoting claims.
- GGUF: exported through an explicit external llama.cpp checkout.

## Release gates

No release may be tagged until CI passes, source-license ownership is approved,
README and model cards are accurate, the external llama.cpp path is verified,
artifact terms are explicit, and the maintainer approves the release checklist.

The next candidate is `v0.1.0-alpha`; it is prepared but not tagged.
