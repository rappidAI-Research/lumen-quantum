from pathlib import Path

import pytest

from scripts.echelon_checkpoint_manifest import (
    build_manifest,
    verify_manifest,
    write_atomic,
)

pytestmark = pytest.mark.unit


def test_checkpoint_manifest_detects_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "trainer_state.json").write_text('{"step":1}', encoding="utf-8")

    manifest = build_manifest(checkpoint, run_id="test-run", processed_tokens=1234)
    assert verify_manifest(checkpoint, manifest) == []

    (checkpoint / "model.safetensors").write_bytes(b"changed")
    problems = verify_manifest(checkpoint, manifest)
    assert any("size mismatch" in problem or "checksum mismatch" in problem for problem in problems)


def test_checkpoint_manifest_is_written_atomically(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model")

    manifest = build_manifest(checkpoint, run_id="run-1", processed_tokens=100)
    output = checkpoint / "checkpoint-manifest.json"
    write_atomic(output, manifest)

    assert output.is_file()
    assert verify_manifest(checkpoint, manifest) == []
