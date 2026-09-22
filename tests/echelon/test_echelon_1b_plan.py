from pathlib import Path

import pytest

from scripts.echelon_run_status import initial_status, load_status, update_status, write_atomic
from scripts.validate_echelon_1b import ROOT, manual_parameter_count, validate, validate_sources

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


def test_source_registry_pins_identified_candidates_without_approving_them() -> None:
    import yaml

    path = ROOT / "configs" / "echelon" / "1b" / "sources.yaml"
    sources = yaml.safe_load(path.read_text(encoding="utf-8"))["sources"]
    by_id = {item["id"]: item for item in sources}

    expected_revisions = {
        "de-fineweb2-hq": "c0c06e94fd3a44ae9e802b2b0fc533817601eb5e",
        "en-fineweb-edu": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "math-finemath": "e92b25a616738fe95dc186b64dfb19f9c8525594",
        "code-stack-edu": "eeec5caac5cc3758a18f1d3ba4416837a9ba814c",
    }
    for source_id, revision in expected_revisions.items():
        assert by_id[source_id]["revision"] == revision
        assert by_id[source_id]["production_approved"] is False
        assert by_id[source_id]["evidence"]["dataset_card"].startswith("https://")


def test_source_production_gate_remains_closed_until_explicit_approval() -> None:
    blockers = validate_sources(require_production_ready=True)
    for source_id in (
        "de-fineweb2-hq",
        "en-fineweb-edu",
        "reference-knowledge",
        "math-finemath",
        "code-stack-edu",
    ):
        assert f"sources.yaml: {source_id} is not production approved" in blockers


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
