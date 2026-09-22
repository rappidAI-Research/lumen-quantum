import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.echelon_shard_stream import ShardedTokenStream, load_manifest

pytestmark = pytest.mark.unit


def _write_shard(path: Path, values: list[int]) -> dict:
    array = np.asarray(values, dtype=np.uint16)
    path.write_bytes(array.tobytes())
    return {
        "path": path.name,
        "tokens": len(values),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _manifest(tmp_path: Path, *, context_length: int = 4) -> Path:
    shards = [
        _write_shard(tmp_path / "train-00000.bin", list(range(0, 6))),
        _write_shard(tmp_path / "train-00001.bin", list(range(6, 14))),
    ]
    payload = {
        "schema_version": "1.0.0",
        "model_line": "quantum-1-echelon",
        "split": "train",
        "token_dtype": "uint16",
        "context_length": context_length,
        "total_tokens": 14,
        "shards": shards,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_stream_crosses_shard_boundary_and_resumes_exactly(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    stream = ShardedTokenStream(manifest, verify_files=True)

    first = stream.next_sequence()
    second = stream.next_sequence()
    assert first.tolist() == [0, 1, 2, 3]
    assert second.tolist() == [4, 5, 6, 7]
    assert stream.cursor.global_token_offset == 8
    assert stream.cursor.sequences_emitted == 2

    resumed = ShardedTokenStream(manifest, start_token_offset=8)
    assert resumed.next_sequence().tolist() == [8, 9, 10, 11]
    assert resumed.cursor.global_token_offset == 12
    assert resumed.remaining_full_sequences == 0


def test_resume_offset_must_be_sequence_aligned(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="align"):
        ShardedTokenStream(manifest, start_token_offset=3)


def test_manifest_rejects_wrong_total(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["total_tokens"] = 15
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="shard sum"):
        load_manifest(manifest)
