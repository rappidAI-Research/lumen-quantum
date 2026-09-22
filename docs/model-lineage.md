# Model lineage

## Historical pilot line

`quantum-1-pilot` is the public legacy/base experiment. Its manifest uses the
historical model ID `quantum-1-base`. It was independently pretrained from
random initialization with a custom tokenizer.

`quantum-1.6-pilot` continues from the pilot weights while preserving the
architecture and approved tokenizer and resetting training state. It is a later
experimental stage, not proof of a generally capable model.

## Strategic Echelon line

`quantum-1-echelon` is the current strategic model line; its public model name is
**Quantum 1 Echelon**. Base is the from-scratch pretraining stage and Chat is the
required project endpoint after supervised and preference post-training. They are
stages of one family and must not be listed as independent model families. The
earlier 506M configuration remains historical preflight evidence; the strategic
production target is now approximately 1B parameters.

The checked-in Echelon path configuration forbids reuse of pilot models,
tokenizers, and data for the current from-scratch design. Future changes to that
design require a documented upstream model, revision, license, tokenizer, and
compatibility review.

## Naming rule

Preserve public model IDs and historical file paths. Use “rappidAI Quantum” for
the project, “Lumen” only for historical context, and qualified stage names when
discussing Echelon.
