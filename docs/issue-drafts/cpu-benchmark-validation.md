# task: capture a real CPU inference benchmark record

**Type:** reproducibility/benchmark · **Good first issue:** no (needs a local
model) · **GPU required:** no

## Problem

`scripts/benchmark_cpu_inference.py` is tested in `--dry-run` mode, but no record
from a real local model has been captured. The repository therefore makes no CPU
speed claims.

## Acceptance criteria

- Run the tool against a local pilot model directory you already have:
  `python scripts/benchmark_cpu_inference.py --model-path <local> --output reports/cpu/<name>.json`.
- Commit the resulting record and, if useful, extend
  [../cpu-benchmarking.md](../cpu-benchmarking.md) with the measurement context.
- Keep the stated limitations; do not present the numbers as capability claims.

## Notes

No model is downloaded; the model directory must already exist locally.
