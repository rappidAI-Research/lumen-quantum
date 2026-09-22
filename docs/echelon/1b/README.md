# Quantum 1 Echelon 1B execution contract

This directory is the repository-side operational companion to the canonical
Quantum 1 Echelon 1B Masterplan v1.2. It converts the plan into versioned
configuration and release gates without claiming that the planned work has been
executed.

## Identity

- Public model name: **Quantum 1 Echelon**.
- Technical model line / codename: `quantum-1-echelon`.
- `quantum-1-echelon-base` and the Chat stage are stages of one model line, not
  separate model families.
- The historical 506M Echelon configuration remains retained as preflight
  evidence and is not overwritten by the 1B work.

## Current target

The Base-stage target is approximately 1.0–1.02B parameters, 4,096-token
context and BF16 training. Candidate A is 32K vocabulary / 21 layers
(~1.014B parameters); candidate B is 48K vocabulary / 20 layers (~1.000B).
Exactly one candidate may be frozen after tokenizer A/B evidence exists.

Base pretraining has a fixed 40B-token target. A 50B extension is conditional:
it is allowed only when measured learning progress and remaining promotional
credits cover the extra Base cost while the protected Chat budget and safety
reserve remain intact.

The project is complete only after the Chat path has been evaluated: supervised
fine-tuning, preference training (or a documented decision to retain the better
SFT checkpoint), Chat evaluation and reproducible export.

## Gate order

1. Freeze architecture, tokenizer and approved source registry.
2. Produce Garden v2 data, stable splits, decontamination evidence, shards,
   checksums and a final manifest.
3. Prove the production shard loader, exact data-position resume, checkpoint
   recovery and SFT/DPO code paths without expensive GPU time.
4. Run a 60–90M-token LR sweep.
5. Start one production run and use 100M as the first numerical/throughput gate.
6. Prove external recovery plus client-disconnect safety early in that same run.
7. Use 1B as the calibration/cost gate, then continue to 5/10/20/30/40B only
   while quality and budget gates remain green.
8. Run Base evaluation, mandatory Chat post-training, Chat evaluation and
   release preparation.

## Paid-compute prohibition

A large H100 session is not authorized merely because these files exist. Before
paid production compute, the maintainer must have current quota/capacity/pricing,
a frozen config and tokenizer, a final data manifest, a production-capable shard
loader, verified checkpoint/resume, unattended execution, and a working Chat
post-training smoke path.

See [STATUS.md](STATUS.md) for the current short state and
[AWS_RUNBOOK.md](AWS_RUNBOOK.md) for the planned execution boundary.
