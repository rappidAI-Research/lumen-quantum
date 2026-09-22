import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.echelon_base_smoke import run_smoke
from scripts.echelon_checkpoint_manifest import verify_manifest

pytestmark = pytest.mark.unit


def _manifest(tmp_path: Path) -> Path:
    values = np.arange(96, dtype=np.uint16) % 200
    shard = tmp_path / "train-00000.bin"
    shard.write_bytes(values.tobytes())
    payload = {
        "schema_version": "1.0.0",
        "model_line": "quantum-1-echelon",
        "split": "train",
        "token_dtype": "uint16",
        "context_length": 16,
        "total_tokens": int(values.size),
        "shards": [
            {
                "path": shard.name,
                "tokens": int(values.size),
                "bytes": shard.stat().st_size,
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_base_smoke_resumes_exact_data_position(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    config = Path("configs/echelon/1b/smoke-tiny.yaml")

    first = run_smoke(
        config_path=config,
        manifest_path=manifest,
        output_dir=tmp_path / "run1",
        run_id="smoke-a",
        max_steps=2,
    )
    assert first["global_step"] == 2
    assert first["processed_tokens"] == 32
    assert first["data_offset"] == 32

    second = run_smoke(
        config_path=config,
        manifest_path=manifest,
        output_dir=tmp_path / "run2",
        run_id="smoke-a",
        max_steps=1,
        resume_dir=Path(first["checkpoint_dir"]),
    )
    assert second["global_step"] == 3
    assert second["processed_tokens"] == 48
    assert second["data_offset"] == 48

    checkpoint = Path(second["checkpoint_dir"])
    payload = json.loads((checkpoint / "checkpoint-manifest.json").read_text(encoding="utf-8"))
    assert verify_manifest(checkpoint, payload) == []
