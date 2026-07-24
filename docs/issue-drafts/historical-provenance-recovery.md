# reproducibility task: recover historical dataset revisions

**Type:** reproducibility · **Good first issue:** no (needs maintainer knowledge)
· **GPU required:** no

## Problem

`reports/provenance/historical-run-provenance-status.json` lists the pilot
pipelines with `dataset_revision: null` and `dataset_revision_status: "unknown"`
because historical configs reference the mutable revision `main`. The exact
immutable dataset revision used by each historical run is not recorded.

## Acceptance criteria

- For each run the maintainer can confirm, record the exact immutable dataset
  revision and its evidence, and update `dataset_revision_status`.
- Do not guess or backfill an unverified revision.
- `python scripts/build_run_manifest.py --validate-provenance reports/provenance/historical-run-provenance-status.json`
  still passes.

## Notes

This documents what is known and keeps unknowns explicit; no training is run.
