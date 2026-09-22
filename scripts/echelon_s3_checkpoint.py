"""Publish verified Quantum 1 Echelon recovery checkpoints to private S3.

The recovery-valid marker is written last. A checkpoint is not externally valid
until every file and its manifest have been uploaded and verified by size plus
SHA-256 metadata.

This module imports boto3 only for the CLI path so unit tests remain offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from scripts.echelon_checkpoint_manifest import sha256_file, verify_manifest


class S3Client(Protocol):
    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        ExtraArgs: dict[str, Any] | None = None,
    ) -> Any: ...

    def put_object(self, **kwargs: Any) -> Any: ...

    def head_object(self, **kwargs: Any) -> dict[str, Any]: ...


def _join_key(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def _verify_remote_object(
    client: S3Client,
    *,
    bucket: str,
    key: str,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    head = client.head_object(Bucket=bucket, Key=key)
    if int(head["ContentLength"]) != expected_bytes:
        raise RuntimeError(f"S3 size mismatch for s3://{bucket}/{key}")
    metadata = {str(k).lower(): str(v) for k, v in head.get("Metadata", {}).items()}
    if metadata.get("sha256") != expected_sha256:
        raise RuntimeError(f"S3 SHA-256 metadata mismatch for s3://{bucket}/{key}")


def publish_checkpoint(
    checkpoint_dir: Path,
    manifest_path: Path,
    *,
    bucket: str,
    prefix: str,
    client: S3Client,
) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = verify_manifest(checkpoint_dir, manifest)
    if problems:
        raise ValueError("local checkpoint verification failed: " + "; ".join(problems))

    run_id = str(manifest["run_id"])
    processed_tokens = int(manifest["processed_tokens"])
    checkpoint_prefix = _join_key(prefix, run_id, "checkpoints", str(processed_tokens))

    for entry in manifest["files"]:
        relative = str(entry["path"])
        source = checkpoint_dir / relative
        key = _join_key(checkpoint_prefix, "files", relative)
        checksum = str(entry["sha256"])
        client.upload_file(
            str(source),
            bucket,
            key,
            ExtraArgs={
                "ServerSideEncryption": "AES256",
                "Metadata": {"sha256": checksum},
            },
        )
        _verify_remote_object(
            client,
            bucket=bucket,
            key=key,
            expected_bytes=int(entry["bytes"]),
            expected_sha256=checksum,
        )

    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_key = _join_key(checkpoint_prefix, "checkpoint-manifest.json")
    client.put_object(
        Bucket=bucket,
        Key=manifest_key,
        Body=manifest_bytes,
        ContentType="application/json",
        ServerSideEncryption="AES256",
        Metadata={"sha256": manifest_sha256},
    )
    _verify_remote_object(
        client,
        bucket=bucket,
        key=manifest_key,
        expected_bytes=len(manifest_bytes),
        expected_sha256=manifest_sha256,
    )

    marker = {
        "schema_version": "1.0.0",
        "model_line": "quantum-1-echelon",
        "run_id": run_id,
        "processed_tokens": processed_tokens,
        "manifest_key": manifest_key,
        "manifest_sha256": manifest_sha256,
    }
    marker_bytes = (json.dumps(marker, sort_keys=True, indent=2) + "\n").encode("utf-8")
    marker_sha256 = hashlib.sha256(marker_bytes).hexdigest()
    marker_key = _join_key(checkpoint_prefix, "_RECOVERY_VALID.json")
    client.put_object(
        Bucket=bucket,
        Key=marker_key,
        Body=marker_bytes,
        ContentType="application/json",
        ServerSideEncryption="AES256",
        Metadata={"sha256": marker_sha256},
    )
    _verify_remote_object(
        client,
        bucket=bucket,
        key=marker_key,
        expected_bytes=len(marker_bytes),
        expected_sha256=marker_sha256,
    )

    return f"s3://{bucket}/{checkpoint_prefix}"


def _boto3_client() -> S3Client:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for S3 publication. Install with: pip install -e '.[cloud]'"
        ) from exc
    return boto3.client("s3")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish a verified Echelon recovery checkpoint to private S3."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="quantum-1-echelon")
    args = parser.parse_args()

    uri = publish_checkpoint(
        args.checkpoint_dir,
        args.manifest,
        bucket=args.bucket,
        prefix=args.prefix,
        client=_boto3_client(),
    )
    print(f"Recovery-valid checkpoint: {uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
