# Reproducibility

## Required run record

Every meaningful run should retain:

- code commit and clean/dirty state;
- complete resolved Python environment and platform;
- exact YAML files and any overrides;
- dataset identifier, configuration, immutable revision, terms, and manifest;
- tokenizer files, vocabulary, special-token IDs, revision, and SHA-256;
- upstream model identifier, revision, and license when applicable;
- seeds, deterministic settings, and split method;
- hardware, runtime, precision, and elapsed time;
- checkpoints with step/epoch and resume lineage;
- raw evaluation outputs and metric implementation; and
- export tool revision, command, file sizes, and checksums.

## Current gaps

The public pilot releases do not provide a complete, versioned chain of final
run manifest, dependency lock, raw logs, hardware/runtime record, raw evaluation
output, tokenizer license, and model-weight license. Their cards must therefore
distinguish verified artifacts from publisher-reported results.

The Echelon data config pins FineWeb2-HQ, but production preparation and model
training are incomplete. Its architecture and tokenizer reports do not establish
trained-model capability.

## Determinism limits

Seeds do not guarantee identical results across operating systems, library
versions, hardware, kernels, thread counts, or distributed execution. Report
the environment and compare artifacts/metrics rather than promising bitwise
identity without evidence.
