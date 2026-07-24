"""Collect a reproducibility run manifest from config, Git and the environment.

The manifest records what is known and leaves everything else ``null``. It never
copies large artifacts into Git, never publishes weights, and never guesses
unknown provenance. ``verification_status`` defaults to ``"unknown"`` and
``completion_status`` to ``"metadata-only"`` so a generated manifest is not
mistaken for a completed training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "run-manifest.schema.json"
SCHEMA_VERSION = "1.1.0"
PROVENANCE_REQUIRED = (
    "model_line",
    "code_revision",
    "configuration_files",
    "dataset_id",
    "dataset_subset",
    "dataset_revision",
    "dataset_revision_status",
    "tokenizer_revision",
    "tokenizer_checksum",
    "model_artifact_revision",
    "model_checksum",
    "seeds",
    "training_steps",
    "training_steps_status",
    "hardware",
    "runtime",
    "final_manifest",
    "raw_logs",
    "raw_evaluation",
    "license_status",
    "verification_status",
    "notes",
)
PROVENANCE_STATUS = {
    "unknown",
    "partial",
    "publisher-reported",
    "configured-target",
    "not-yet-completed",
    "verified",
}
RECORDED_PACKAGES = (
    "torch",
    "transformers",
    "tokenizers",
    "sentencepiece",
    "datasets",
    "accelerate",
    "numpy",
    "PyYAML",
)
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
_UTC = timezone.utc  # noqa: UP017 (Python 3.10 runtime fallback; project targets 3.11+)


def _now() -> str:
    return datetime.now(_UTC).isoformat()


def _git(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in RECORDED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _lookup(mapping: Any, *keys: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _config_entries(config_paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    merged: dict[str, Any] = {}
    for path in config_paths:
        try:
            rel = str(path.resolve().relative_to(ROOT))
        except ValueError:
            rel = str(path)
        if path.is_file():
            entries.append({"path": rel, "sha256": sha256_file(path)})
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                merged.update(loaded)
        else:
            entries.append({"path": rel, "sha256": None, "note": "missing"})
    return entries, merged


def _record_artifacts(artifacts: list[Path]) -> list[dict[str, Any]]:
    recorded: list[dict[str, Any]] = []
    for artifact in artifacts:
        if not artifact.is_file():
            recorded.append(
                {"name": str(artifact), "size_bytes": None, "sha256": None, "note": "missing"}
            )
            continue
        size = artifact.stat().st_size
        recorded.append(
            {
                "name": artifact.name,
                "path": str(artifact),
                "size_bytes": size,
                "sha256": sha256_file(artifact) if size <= MAX_ARTIFACT_BYTES else None,
            }
        )
    return recorded


def build_manifest(
    config_paths: list[Path],
    artifacts: list[Path] | None = None,
    completion_status: str = "metadata-only",
    error_status: str | None = None,
) -> dict[str, Any]:
    started = _now()
    config_files, merged = _config_entries(config_paths)
    dataset = merged.get("dataset")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "tool": "build_run_manifest.py",
        "code": {
            "revision": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "configuration_files": config_files,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
        },
        "dependencies": _package_versions(),
        "seeds": _lookup(merged, "seed", "seeds"),
        "dataset": {
            "id": _lookup(dataset, "id", "path", "name"),
            "subset": _lookup(dataset, "subset"),
            "revision": _lookup(dataset, "revision"),
        },
        "model": {"artifact_revision": None, "checksum": None},
        "tokenizer": {"revision": None, "checksum": None},
        "artifacts": _record_artifacts(artifacts or []),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "note": "Completed by the maintainer for real runs; never guessed here.",
        },
        "timing": {"started_at": started, "finished_at": _now()},
        "completion_status": completion_status,
        "error_status": error_status,
        "verification_status": "unknown",
    }
    return manifest


def load_schema() -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def validate_against_schema(obj: dict[str, Any], schema: dict[str, Any] | None = None) -> list[str]:
    schema = schema or load_schema()
    errors: list[str] = []
    required = schema.get("required", [])
    if isinstance(required, list):
        errors.extend(f"missing required field: {key}" for key in required if key not in obj)
    properties = schema.get("properties", {})
    for field in ("verification_status", "completion_status"):
        enum = properties.get(field, {}).get("enum")
        if enum and obj.get(field) not in enum:
            errors.append(f"invalid {field}: {obj.get(field)!r}")
    return errors


def validate_provenance_entry(entry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    errors.extend(
        f"missing provenance field: {key}" for key in PROVENANCE_REQUIRED if key not in entry
    )
    status = entry.get("verification_status")
    if status not in PROVENANCE_STATUS:
        errors.append(f"invalid verification_status: {status!r}")
    for field in ("dataset_revision_status", "training_steps_status"):
        value = entry.get(field)
        if value is not None and value not in PROVENANCE_STATUS:
            errors.append(f"invalid {field}: {value!r}")
    return errors


def _validate_provenance(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload if isinstance(payload, list) else [payload]
    problems: list[str] = []
    for index, entry in enumerate(entries):
        problems.extend(
            f"{path.name}[{index}]: {error}" for error in validate_provenance_entry(entry)
        )
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        return 1
    print(f"Validated {len(entries)} provenance entry(ies).")
    return 0


def _validate_file(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload if isinstance(payload, list) else [payload]
    schema = load_schema()
    problems: list[str] = []
    for index, entry in enumerate(entries):
        problems.extend(
            f"{path.name}[{index}]: {error}" for error in validate_against_schema(entry, schema)
        )
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        return 1
    print(f"Validated {len(entries)} manifest entry(ies) against the schema.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate a reproducibility run manifest."
    )
    parser.add_argument(
        "--config", type=Path, action="append", default=[], help="YAML config (repeatable)."
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        action="append",
        default=[],
        help="Local artifact to checksum in place.",
    )
    parser.add_argument("--completion-status", default="metadata-only")
    parser.add_argument("--error-status", default=None)
    parser.add_argument(
        "--output", type=Path, help="Write the manifest JSON here instead of stdout."
    )
    parser.add_argument(
        "--validate", type=Path, help="Validate an existing run manifest against the schema."
    )
    parser.add_argument(
        "--validate-provenance",
        type=Path,
        help="Validate a historical provenance registry against its required fields.",
    )
    arguments = parser.parse_args()
    if arguments.validate_provenance is not None:
        return _validate_provenance(arguments.validate_provenance)
    if arguments.validate is not None:
        return _validate_file(arguments.validate)
    if not arguments.config:
        parser.error("either --config or --validate is required")
    manifest = build_manifest(
        list(arguments.config),
        list(arguments.artifact),
        completion_status=arguments.completion_status,
        error_status=arguments.error_status,
    )
    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    if arguments.output is not None:
        arguments.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote manifest to {arguments.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
