import json
from pathlib import Path
from typing import Any

import pytest

from scripts.echelon_checkpoint_manifest import build_manifest, write_atomic
from scripts.echelon_s3_checkpoint import publish_checkpoint

pytestmark = pytest.mark.unit


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.order: list[str] = []

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        ExtraArgs: dict[str, Any] | None = None,
    ) -> None:
        body = Path(filename).read_bytes()
        args = ExtraArgs or {}
        self.objects[(bucket, key)] = {
            "Body": body,
            "Metadata": dict(args.get("Metadata", {})),
        }
        self.order.append(key)

    def put_object(self, **kwargs: Any) -> None:
        body = kwargs["Body"]
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = {
            "Body": bytes(body),
            "Metadata": dict(kwargs.get("Metadata", {})),
        }
        self.order.append(str(kwargs["Key"]))

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        item = self.objects[(kwargs["Bucket"], kwargs["Key"])]
        return {
            "ContentLength": len(item["Body"]),
            "Metadata": item["Metadata"],
        }


def test_recovery_marker_is_published_last(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "trainer_state.json").write_text('{"step":1}', encoding="utf-8")

    payload = build_manifest(checkpoint, run_id="run-a", processed_tokens=1000)
    manifest_path = checkpoint / "checkpoint-manifest.json"
    write_atomic(manifest_path, payload)

    client = FakeS3()
    uri = publish_checkpoint(
        checkpoint,
        manifest_path,
        bucket="private-test",
        prefix="echelon",
        client=client,
    )

    assert uri == "s3://private-test/echelon/run-a/checkpoints/1000"
    assert client.order[-1].endswith("_RECOVERY_VALID.json")
    marker = json.loads(client.objects[("private-test", client.order[-1])]["Body"])
    assert marker["run_id"] == "run-a"
    assert marker["processed_tokens"] == 1000


def test_publication_refuses_mutated_local_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = checkpoint / "model.safetensors"
    model.write_bytes(b"model")

    payload = build_manifest(checkpoint, run_id="run-b", processed_tokens=2000)
    manifest_path = checkpoint / "checkpoint-manifest.json"
    write_atomic(manifest_path, payload)
    model.write_bytes(b"mutated")

    with pytest.raises(ValueError, match="local checkpoint verification failed"):
        publish_checkpoint(
            checkpoint,
            manifest_path,
            bucket="private-test",
            prefix="echelon",
            client=FakeS3(),
        )
