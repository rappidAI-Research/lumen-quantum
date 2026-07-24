# good first issue: add a config-invariant validation test

**Type:** test · **Good first issue:** yes · **GPU required:** no

## Problem

The README documents configuration invariants (for example the pilot
architecture reaching 49,295,872 parameters at `vocab_size: 16384`). Some
invariants are not yet asserted by an automated test, so drift could go
unnoticed.

## Acceptance criteria

- Add a `unit`-marked test under `tests/` that asserts one concrete, documented
  configuration invariant.
- The test must run without a GPU, network, or the optional ML dependencies.
- `python -m pytest -m "unit and not slow and not gpu and not network"` passes.

## Notes

Keep the test small and deterministic. See `tests/test_repository_health.py`.
