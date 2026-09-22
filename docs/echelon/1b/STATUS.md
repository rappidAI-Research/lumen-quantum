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
- Tokenizer A/B planning, Garden v2 planning, source registry and Base/SFT/DPO
  execution contracts.
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

## Open decisions before freeze

1. 32K/21-layer vs 48K/20-layer tokenizer/model candidate.
2. RoPE base after a bounded local/cheap ablation.
3. Exact immutable revisions and rights/terms review for every Garden v2 source.
4. Final Base learning-rate schedule after the short LR sweep.
5. Final SFT and preference datasets/hyperparameters after Base evaluation.
6. AWS region and Spot/On-Demand choice after quota, capacity and price checks.

## Next action

Add the systemd deployment interface and wire the unattended guard, Base smoke,
checkpoint integrity and S3 publisher into one end-to-end disconnect/recovery
fixture. Then implement production chat-data validators and freeze the tokenizer
A/B experiment inputs. No large H100 session is allowed before those recovery,
data and post-training gates are green.
