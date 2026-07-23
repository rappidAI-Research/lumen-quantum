"""Fail fast on repository states that are unsafe or misleading for contributors."""

from __future__ import annotations

import configparser
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 10 * 1024 * 1024
FORBIDDEN_ARTIFACT_SUFFIXES = {".bin", ".gguf", ".pt", ".pth", ".safetensors"}
FORBIDDEN_TRACKED_NAMES = {".env", ".env.local"}


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def tracked_files() -> list[Path]:
    return [ROOT / item for item in git_output("ls-files").splitlines() if item]


def registered_submodule_paths() -> set[str]:
    metadata = ROOT / ".gitmodules"
    if not metadata.exists():
        return set()
    parser = configparser.ConfigParser()
    parser.read(metadata, encoding="utf-8")
    return {
        parser[section]["path"]
        for section in parser.sections()
        if parser.has_option(section, "path")
    }


def unregistered_gitlinks() -> list[str]:
    gitlinks = {
        line.split(maxsplit=3)[3]
        for line in git_output("ls-files", "-s").splitlines()
        if line.startswith("160000 ")
    }
    return sorted(gitlinks - registered_submodule_paths())


def health_errors() -> list[str]:
    errors: list[str] = []
    missing_metadata = unregistered_gitlinks()
    if missing_metadata:
        errors.append(
            "Git links without matching .gitmodules entries: " + ", ".join(missing_metadata)
        )

    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if path.name in FORBIDDEN_TRACKED_NAMES:
            errors.append(f"Sensitive environment file is tracked: {relative}")
        if path.suffix.lower() in FORBIDDEN_ARTIFACT_SUFFIXES:
            errors.append(f"Generated model/data artifact is tracked: {relative}")
        if path.is_file() and path.stat().st_size > MAX_TRACKED_BYTES:
            errors.append(f"Tracked file exceeds 10 MiB: {relative}")
    return errors


def main() -> int:
    errors = health_errors()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Repository health checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
