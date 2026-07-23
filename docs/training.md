# Training and resume

## From-scratch pilots

The historical smoke and `quantum-1-pilot` designs construct
`LlamaForCausalLM` from an explicit `LlamaConfig` and random weights. Reproducing
those experiments must not substitute an external pretrained model.

## Continued pilot

`quantum-1.6-pilot` is configured for weights-only initialization from the local
pilot checkpoint. The preflight requires the same architecture and approved
tokenizer, then resets optimizer, scheduler, and global step. The configured
30,518 steps are a target derived from 500M additional tokens and 16,384 tokens
per optimizer step; they are not proof of a completed run.

## Echelon

Echelon Base is a configured 506,333,440-parameter causal decoder. No production
dataset or trained checkpoint is public. Echelon Chat is a future stage inside
the same model line.

## Resume security

Local resume files include optimizer/scheduler and RNG state loaded with
`weights_only=False`. Pickle-compatible data can execute code. Resume only from
a trusted local run or an artifact with independently verified provenance and
checksum. Prefer safetensors for standalone model weights.

## Future derived models

Fine-tuning is allowed only with recorded upstream model ID, immutable revision,
license/terms, tokenizer, architecture changes, data terms, and release-license
compatibility. The pilot from-scratch rule does not silently prohibit all future
adaptation.
