# Quantum 1 Echelon 1B AWS runbook (pre-production)

Status: design/runbook only. It does not prove that AWS resources, quotas,
capacity or buckets exist.

## Principle

The MacBook is a control and observation client only. The authoritative training
process must live on AWS and must continue through SSH disconnects, local
Internet loss and laptop sleep.

## Planned runtime boundary

- Preferred accelerator class: one H100 80 GB (`p5.4xlarge`) after current
  regional price, quota and capacity checks.
- Production process lifecycle: server-side `systemd` service or equivalent
  supervisor. `tmux` may be used for observation, never as the only lifecycle
  mechanism.
- Active dataset cache and temporary checkpoints: local instance NVMe.
- Canonical shards, manifests and verified recovery checkpoints: private S3 in
  the chosen training region.
- No interactive prompts after preflight validation succeeds.

## Required helpers before production

The following interfaces are required before the paid Base run:

- `start`: validate frozen configs/manifests, then start the supervised service;
- `status`: show service state plus the machine-readable run-status record;
- `logs`: follow persistent server-side logs after reconnect;
- `stop`: request a controlled checkpoint and shutdown;
- `resume-latest`: locate the newest verified external recovery checkpoint and
  restore model, optimizer, scheduler, RNG and exact data position;
- `sync-checkpoint`: hash local checkpoint, upload with retry/backoff, verify the
  external object and only then mark it recovery-valid.

Names may change during implementation, but the behavior may not be dropped.

## Machine-readable status

Use `schemas/echelon-run-status.schema.json` and atomic writes. States are:
`RUNNING`, `CHECKPOINTING`, `INTERRUPTED`, `FAILED`, `COMPLETED`.
The record must include step, processed tokens, latest checkpoint and latest
verified S3 sync.

## Mandatory failure tests

Before substantial H100 spend:

1. Start a bounded training smoke through the supervisor.
2. Disconnect SSH and take the local client offline; confirm training continues.
3. Produce and verify an external recovery checkpoint.
4. Terminate the process/Spot instance in a controlled test.
5. Launch replacement capacity and run `resume-latest`.
6. Confirm optimizer, scheduler, RNG, step, processed-token count and shard/offset
   continue correctly.

Any failure blocks the large production run.

## Cost safety

The committed planning ceiling is $1,140 promotional credits with no planned
private overage. Runtime tooling must therefore include wall-time bounds,
shutdown on completion/failure and an idle-instance watchdog. AWS Budgets are
alerts, not a hard kill switch.
