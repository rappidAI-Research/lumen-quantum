from pathlib import Path

import pytest

from scripts.echelon_run_status import initial_status, load_status, update_status, write_atomic
from scripts.validate_echelon_1b import ROOT, manual_parameter_count, validate

pytestmark = pytest.mark.unit


def test_echelon_1b_planning_contract() -> None:
    assert validate() == []


def test_model_candidates_match_masterplan_counts() -> None:
    import yaml

    expected = {
        "model-32k.yaml": 1_014_061_056,
        "model-48k.yaml": 1_000_163_328,
    }
    for name, parameter_count in expected.items():
        path = ROOT / "configs" / "echelon" / "1b" / name
        model = yaml.safe_load(path.read_text(encoding="utf-8"))["model"]
        assert manual_parameter_count(model) == parameter_count


def test_status_file_is_atomic_and_resume_relevant(tmp_path: Path) -> None:
    path = tmp_path / "run-status.json"
    status = initial_status("test-run")
    status = update_status(
        status,
        state="CHECKPOINTING",
        step=123,
        processed_tokens=100_000_000,
        last_checkpoint="checkpoint-100m",
        last_verified_s3_sync="s3://example/checkpoint-100m",
        message="checkpoint verified",
    )
    write_atomic(path, status)
    loaded = load_status(path)
    assert loaded["state"] == "CHECKPOINTING"
    assert loaded["step"] == 123
    assert loaded["processed_tokens"] == 100_000_000
    assert loaded["last_checkpoint"] == "checkpoint-100m"
    assert loaded["last_verified_s3_sync"] == "s3://example/checkpoint-100m"
