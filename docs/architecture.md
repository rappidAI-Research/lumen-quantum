# Architecture

## System boundary

This repository owns versioned ML pipeline code, configuration, tests, and
small evidence reports. The website renders research information but does not
implement the pipelines. Hugging Face hosts separately governed release
artifacts.

## Pipeline layers

1. **Configuration:** YAML records paths, revisions, seeds, filters, budgets,
   architecture, optimizer, checkpoint, evaluation, and export settings.
2. **Data:** streamed or local records are normalized, filtered, deduplicated,
   split by stable hashes, tokenized, and summarized in manifests.
3. **Tokenizer:** SentencePiece artifacts are trained locally, validated for
   round-trip/special-token behavior, and identified by checksums.
4. **Model:** explicit `LlamaConfig` values construct random or meta-device
   models; pilot continuation loads approved local weights only.
5. **Training:** Accelerate coordinates CPU/CUDA execution, checkpoints, resume,
   loss evaluation, and sample generations.
6. **Evaluation/export:** fixed completion prompts, loss/perplexity, and an
   explicit external llama.cpp checkout produce reviewable outputs.

## Artifact boundary

`configs/`, `scripts/`, `tests/`, `docs/`, small `reports/`, and fixed eval
prompts are tracked. Raw data, tokenized tensors, tokenizer binaries, model
weights, GGUF files, checkpoints, logs, and generated reports are ignored.

## Trust boundary

YAML and JSON are parsed with safe loaders. External datasets, model weights,
tokenizers, resume checkpoints, executables, and conversion scripts are
untrusted inputs. Resume state uses pickle-compatible loading and must only be
opened when created locally or verified through a trusted provenance chain.
