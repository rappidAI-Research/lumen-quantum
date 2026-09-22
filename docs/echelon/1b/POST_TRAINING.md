# Quantum 1 Echelon Chat-stage contract

The Base checkpoint is not the end product. A useful Chat-stage candidate is a
mandatory deliverable of the current plan.

## SFT

Target 50–200M high-quality instruction tokens, German-first with enough English,
Math and Code coverage to protect Base capabilities. Dataset provenance, terms,
deduplication and a held-out validation split are release gates. The chat special
tokens are frozen with the tokenizer before Base training; no late embedding
resize is planned.

## Preference stage

DPO is the initial planned method. It is not allowed to overwrite the SFT result
as "better" by default. The preference checkpoint must beat or at least preserve
the SFT candidate on the agreed Chat evaluation; otherwise the SFT checkpoint
remains the release candidate.

## Evaluation

At minimum compare Base, SFT and preference checkpoints for German language
quality, instruction following, formatting, factual robustness, Math, Code,
repetition/degeneration, English regressions and bounded safety diagnostics.
Raw outputs and exact settings must be retained.
