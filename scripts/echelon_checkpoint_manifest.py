"""Create and verify local checkpoint manifests before remote recovery sync."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"
DEFAULT_EXCLUDES = {"checkpoint-manifest.json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in DEFAULT_EXCLUDES:
            continue
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise ValueError("checkpoint directory contains no files")
    return entries


def build_manifest(root: Path, *, run_id: str, processed_tokens: int) -> dict[str, Any]:
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if processed_tokens < 0:
        raise ValueError("processed_tokens must be non-negative")
    entries = collect_files(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_line": "quantum-1-echelon",
        "run_id": run_id,
        "processed_tokens": processed_tokens,
        "created_at": datetime.now(UTC).isoformat(),
        "files": entries,
    }


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def verify_manifest(root: Path, manifest: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append("unsupported schema_version")
    if manifest.get("model_line") != "quantum-1-echelon":
        problems.append("model_line mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        problems.append("manifest contains no files")
        return problems

    for entry in files:
        relative = str(entry["path"])
        path = root / relative
        if not path.is_file():
            problems.append(f"missing file: {relative}")
            continue
        if path.stat().st_size != int(entry["bytes"]):
            problems.append(f"size mismatch: {relative}")
            continue
        if sha256_file(path) != str(entry["sha256"]):
            problems.append(f"checksum mismatch: {relative}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/verify Echelon checkpoint manifests.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--processed-tokens", type=int)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()

    if args.verify is not None:
        payload = json.loads(args.verify.read_text(encoding="utf-8"))
        problems = verify_manifest(args.checkpoint_dir, payload)
        if problems:
            for problem in problems:
                print(f"ERROR: {problem}")
            return 1
        print("Checkpoint manifest verification: OK")
        return 0

    if args.run_id is None or args.processed_tokens is None:
        parser.error("--run-id and --processed-tokens are required when building a manifest")
    output = args.output or (args.checkpoint_dir / "checkpoint-manifest.json")
    payload = build_manifest(
        args.checkpoint_dir,
        run_id=args.run_id,
        processed_tokens=args.processed_tokens,
    )
    write_atomic(output, payload)
    print(f"Wrote checkpoint manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
