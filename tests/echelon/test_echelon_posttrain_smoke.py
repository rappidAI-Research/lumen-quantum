from pathlib import Path

import pytest

from scripts.echelon_posttrain_smoke import run_dpo_smoke, run_sft_smoke

pytestmark = pytest.mark.unit


def test_sft_smoke_is_finite() -> None:
    loss = run_sft_smoke(Path("configs/echelon/1b/smoke-tiny.yaml"))
    assert loss > 0


def test_dpo_smoke_is_finite() -> None:
    loss = run_dpo_smoke(Path("configs/echelon/1b/smoke-tiny.yaml"))
    assert loss > 0
