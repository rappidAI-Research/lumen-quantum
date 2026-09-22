"""Atomic status file helper for unattended Quantum 1 Echelon runs."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

SCHEMA_VERSION = "1.0.0"
STATES = {"RUNNING", "CHECKPOINTING", "INTERRUPTED", "FAILED", "COMPLETED"}
DEFAULT_STATUS_PATH = Path("logs/quantum-1-echelon/run-status.json")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_status(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("status file must contain a JSON object")
    return cast("dict[str, Any]", payload)


def validate_status(status: dict[str, Any]) -> None:
    if status.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported status schema_version")
    if status.get("state") not in STATES:
        raise ValueError(f"invalid state: {status.get('state')!r}")
    if int(status.get("step", -1)) < 0:
        raise ValueError("step must be >= 0")
    if int(status.get("processed_tokens", -1)) < 0:
        raise ValueError("processed_tokens must be >= 0")
    if not str(status.get("run_id", "")).strip():
        raise ValueError("run_id must be non-empty")


def write_atomic(path: Path, status: dict[str, Any]) -> None:
    validate_status(status)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(status, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def initial_status(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "state": "RUNNING",
        "updated_at": now_iso(),
        "step": 0,
        "processed_tokens": 0,
        "last_checkpoint": None,
        "last_verified_s3_sync": None,
        "message": "initialized",
    }


def update_status(
    status: dict[str, Any],
    *,
    state: str | None = None,
    step: int | None = None,
    processed_tokens: int | None = None,
    last_checkpoint: str | None = None,
    last_verified_s3_sync: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    updated = dict(status)
    if state is not None:
        updated["state"] = state
    if step is not None:
        updated["step"] = step
    if processed_tokens is not None:
        updated["processed_tokens"] = processed_tokens
    if last_checkpoint is not None:
        updated["last_checkpoint"] = last_checkpoint
    if last_verified_s3_sync is not None:
        updated["last_verified_s3_sync"] = last_verified_s3_sync
    if message is not None:
        updated["message"] = message
    updated["updated_at"] = now_iso()
    validate_status(updated)
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage an atomic Echelon unattended-run status file."
    )
    parser.add_argument("--path", type=Path, default=DEFAULT_STATUS_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--run-id", required=True)

    set_parser = subparsers.add_parser("set")
    set_parser.add_argument("--state", choices=sorted(STATES))
    set_parser.add_argument("--step", type=int)
    set_parser.add_argument("--processed-tokens", type=int)
    set_parser.add_argument("--last-checkpoint")
    set_parser.add_argument("--last-verified-s3-sync")
    set_parser.add_argument("--message")

    subparsers.add_parser("show")
    args = parser.parse_args()

    if args.command == "init":
        status = initial_status(args.run_id)
        write_atomic(args.path, status)
    elif args.command == "set":
        current = load_status(args.path)
        status = update_status(
            current,
            state=args.state,
            step=args.step,
            processed_tokens=args.processed_tokens,
            last_checkpoint=args.last_checkpoint,
            last_verified_s3_sync=args.last_verified_s3_sync,
            message=args.message,
        )
        write_atomic(args.path, status)
    else:
        status = load_status(args.path)
        validate_status(status)

    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
