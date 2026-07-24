# good first issue: align historical example paths

**Type:** documentation · **Good first issue:** yes · **GPU required:** no

## Problem

Several historical guides still use `cd /workspace/LumenQuantum` (for example in
`docs/cloud_pilot.md`, `docs/data_pipeline.md`, `docs/pilot_training_data.md`,
`docs/quantum_1_6_diagnosis.md`, `docs/quantum_1_6_pilot.md`, and
`docs/quantum_1_final_plan.md`). The canonical repository is now
`rappidAI-Research/lumen-quantum`, so this legacy directory name can confuse new
readers.

## Acceptance criteria

- Replace the legacy directory name with a neutral example path (for example
  `cd lumen-quantum`) or add a short "historical example" banner where the
  original wording matters.
- Do not change the technical steps themselves.
- `python scripts/check_markdown_links.py` and
  `python scripts/check_canonical_urls.py` still pass.

## Notes

No GPU, network, or ML dependencies are needed.
