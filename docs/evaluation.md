# Evaluation

The repository supports validation loss/perplexity and fixed completion prompts.
These are infrastructure checks and limited model diagnostics, not a broad
benchmark program.

## Evidence levels

- **Verified raw result:** versioned output linked to exact code/model/data.
- **Publisher-reported result:** stated in a release card without complete raw
  output in this repository.
- **Configured target:** a value in YAML or code, not a measured result.
- **Unknown/incomplete:** required evidence is unavailable.

`quantum-1.6-pilot` loss 3.348852 and perplexity 28.4700 are currently
publisher-reported. No standardized downstream benchmark, uncertainty analysis,
or raw versioned evaluation bundle is published. Echelon has no trained model to
evaluate.

## Publishing results

Record model/checkpoint checksum, tokenizer, prompts/dataset revision, metric
implementation, generation parameters, hardware/runtime, raw outputs, failure
cases, and aggregation. Do not compare systems unless the protocol is genuinely
equivalent.
