# Quantum 1 Echelon 1B status

This file is the short operational control record for the current `quantum-1-echelon`
work. It does not replace run manifests or raw evidence.

## Current phase

**Phase 0 — architecture/tokenizer/data-source freeze: IN PROGRESS**

No 1B production dataset, paid GPU run, Base checkpoint, SFT checkpoint or
preference-trained checkpoint exists yet.

## Fixed project constraints

- Public model name: **Quantum 1 Echelon**.
- Technical model line / codename: `quantum-1-echelon`.
- End product: a useful Chat-stage model; Base is a required intermediate stage.
- AWS promotional-credit ceiling: **$1,140**; no private overage is planned.
- Base target: **40B high-quality tokens**; 50B is conditional only.
- Context: 4,096 for Base pretraining.
- Precision: BF16 for the first production path.
- Preferred compute class: 1x H100 80 GB (`p5.4xlarge`), subject to quota,
  capacity, current pricing and calibration.
- Production execution must be unattended and client-disconnect-safe.

## AWS readiness

- Credit expiry: confirmed by the maintainer to be in 2027.
- AWS cost monitoring: configured by the maintainer.
- P-family GPU quota increases: requested in multiple US regions; approval and
  actual capacity are not yet recorded as verified repository evidence.
- No EC2 GPU instance, production S3 bucket or paid training workload is claimed
  by this file.

## Implemented foundation

- 32K/21-layer and 48K/20-layer approximately-1B architecture candidates.
- Reproducible 32K/48K tokenizer A/B tooling with one shared-corpus identity,
  artifact/config SHA-256 manifests, reserved special-token ID checks, fixed
  domain-tagged evaluation evidence and comparison reports. The real shared
  production tokenizer corpus is not assembled yet and no candidate is frozen.
- Garden v2 planning plus a machine-readable source-readiness gate. Immutable
  candidate revisions and public terms/provenance evidence are captured for
  FineWeb2-HQ, FineWeb-Edu, FineMath-4+ and Stack-Edu, while every source remains
  explicitly unapproved for production pending the required human terms/rights
  decision and removal workflow.
- Base/SFT/DPO execution contracts.
- AWS runtime/budget contract and unattended-operation runbook.
- Machine-readable run status with atomic updates.
- Deterministic memory-mapped uint16 shard stream with exact sequence-aligned
  resume offsets.
- Local checkpoint integrity manifests with SHA-256 verification before remote
  recovery sync is allowed.
- Verified private-S3 checkpoint publication that writes a recovery-valid marker
  only after all checkpoint objects and the manifest pass remote verification.
- Non-interactive process guard with persistent log/status, signal handling and
  hard wall-time enforcement for server-side supervised execution.
- Bounded Base-training smoke path that exercises model/optimizer/scheduler/RNG
  checkpoints and resumes the exact token offset from canonical uint16 shards.
- CPU-safe SFT and DPO optimization smokes that prove post-training mechanics
  without pretending synthetic test tokens are production chat data.
- Versioned systemd service template and strict non-shell entrypoint for
  client-disconnect-safe server-side execution.

## Open decisions before freeze

1. 32K/21-layer vs 48K/20-layer tokenizer/model candidate.
2. RoPE base after a bounded local/cheap ablation.
3. Final Garden v2 source approval: reference/knowledge source selection plus
   human rights/terms/removal decisions for every source.
4. Final Base learning-rate schedule after the short LR sweep.
5. Final SFT and preference datasets/hyperparameters after Base evaluation.
6. AWS region and Spot/On-Demand choice after quota, capacity and price checks.

## Next action

Next, implement the deterministic shared tokenizer-corpus assembler so both
32K/48K candidates can consume identical approved source bytes, including a
production mode that refuses any source whose registry gate is still closed.
Then complete the real tokenizer A/B evidence once the necessary source
approvals/data are available. In parallel, keep production chat-data validation
and the bounded disconnect/recovery acceptance path ready. No large H100 session
is allowed before recovery, tokenizer, source-review and post-training gates are
green.
