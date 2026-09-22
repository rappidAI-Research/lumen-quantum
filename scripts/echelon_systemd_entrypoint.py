"""Systemd entrypoint for unattended Quantum 1 Echelon execution.

The environment file contains a JSON argv array. No shell evaluation is used.
AWS credentials must come from the EC2 instance role, never from this file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.echelon_unattended import run_guarded

REQUIRED_KEYS = {
    "ECHELON_RUN_ID",
    "ECHELON_STATUS_PATH",
    "ECHELON_LOG_PATH",
    "ECHELON_MAX_SECONDS",
    "ECHELON_CHILD_JSON",
}


def load_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)

    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if key in values:
            raise ValueError(f"{path}:{line_number}: duplicate key {key}")
        values[key] = value.strip()

    missing = sorted(REQUIRED_KEYS.difference(values))
    if missing:
        raise ValueError(f"missing required environment keys: {', '.join(missing)}")
    unexpected = sorted(set(values).difference(REQUIRED_KEYS))
    if unexpected:
        raise ValueError("unexpected environment keys are forbidden: " + ", ".join(unexpected))
    return values


def parse_child_command(raw: str) -> list[str]:
    payload = json.loads(raw)
    if not isinstance(payload, list) or not payload:
        raise ValueError("ECHELON_CHILD_JSON must be a non-empty JSON array")
    command: list[str] = []
    for index, item in enumerate(payload):
        if not isinstance(item, str) or not item:
            raise ValueError(f"ECHELON_CHILD_JSON[{index}] must be a non-empty string")
        if "\x00" in item:
            raise ValueError("ECHELON_CHILD_JSON may not contain NUL bytes")
        command.append(item)
    return command


def run_from_env(path: Path) -> int:
    values = load_env_file(path)
    max_seconds = int(values["ECHELON_MAX_SECONDS"])
    if max_seconds <= 0:
        raise ValueError("ECHELON_MAX_SECONDS must be positive")

    return run_guarded(
        parse_child_command(values["ECHELON_CHILD_JSON"]),
        run_id=values["ECHELON_RUN_ID"],
        status_path=Path(values["ECHELON_STATUS_PATH"]),
        log_path=Path(values["ECHELON_LOG_PATH"]),
        max_seconds=max_seconds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch an Echelon systemd-managed run.")
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    return run_from_env(args.env_file)


if __name__ == "__main__":
    raise SystemExit(main())
