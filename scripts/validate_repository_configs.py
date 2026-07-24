"""Parse every tracked YAML, JSON and JSONL configuration or report."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def tracked_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / item for item in result.stdout.splitlines() if item]


def parse_jsonl(path: Path) -> list[Any]:
    records: list[Any] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.relative_to(ROOT)}:{line_number}: {error}") from error
    return records


def validate() -> list[str]:
    validated: list[str] = []
    for path in tracked_paths():
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"} or path.name == "CITATION.cff":
            yaml.safe_load(path.read_text(encoding="utf-8"))
        elif suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif suffix == ".jsonl":
            parse_jsonl(path)
        else:
            continue
        validated.append(str(path.relative_to(ROOT)))
    return validated


def main() -> int:
    validated = validate()
    print(f"Parsed {len(validated)} tracked YAML, JSON, CFF and JSONL files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
