"""Read-only release preflight for an honest source-only ``v0.1.0-alpha``.

Never mutates the repository, never tags, and never publishes. Aggregates
existing repository checks and adds release guards: version consistency across
metadata and release notes, required license and human-approval files, and a
deny-list of unsupported public claims (open weight, trained/released Echelon).
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from scripts.check_canonical_urls import non_canonical_references
from scripts.check_markdown_links import broken_links
from scripts.check_secrets import findings
from scripts.repository_health import health_errors
from scripts.validate_repository_configs import validate

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "README.md",
)
APPROVAL_FILES = (
    "docs/maintainer-source-license-approval.md",
    "MODEL_LICENSES.md",
    "docs/hugging-face/publication-checklist.md",
)
RELEASE_NOTES = "docs/releases/v0.1.0-alpha.md"
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:a|b|rc)\d+$")
BANNED_CLAIMS = (
    "trained echelon model released",
    "echelon model is released",
    "echelon has been trained",
    "echelon training is complete",
    "echelon training completed",
    "quantum-1-echelon is released",
    "is open weight",
    "are open weight",
    "is open-weight",
    "are open-weight",
)


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def tracked_text_files() -> list[Path]:
    paths: list[Path] = []
    for item in git_output("ls-files").splitlines():
        if not item:
            continue
        path = ROOT / item
        if path.suffix.lower() in {".md", ".cff", ".txt", ".toml"} and path.is_file():
            paths.append(path)
    return paths


def project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', text)
    if match is None:
        raise ValueError("pyproject.toml is missing a string project.version")
    return match.group(1)


def clean_worktree_errors() -> list[str]:
    if git_output("status", "--porcelain").strip():
        return ["Git working tree is not clean; commit or stash changes before release."]
    return []


def required_file_errors() -> list[str]:
    errors = [
        f"Required release file is missing: {name}"
        for name in REQUIRED_FILES
        if not (ROOT / name).exists()
    ]
    errors += [
        f"Required approval file is missing: {name}"
        for name in APPROVAL_FILES
        if not (ROOT / name).exists()
    ]
    return errors


def version_errors() -> list[str]:
    errors: list[str] = []
    version = project_version()
    if not VERSION_PATTERN.match(version):
        errors.append(f"pyproject version '{version}' is not a pre-release form like 0.1.0a0.")
    if "0.1.0" not in (ROOT / "CITATION.cff").read_text(encoding="utf-8"):
        errors.append("CITATION.cff does not reference version 0.1.0.")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if "Unreleased" not in changelog and "0.1.0" not in changelog:
        errors.append("CHANGELOG.md has neither an Unreleased nor a 0.1.0 section.")
    notes = ROOT / RELEASE_NOTES
    if not notes.exists():
        errors.append(f"Release notes are missing: {RELEASE_NOTES}")
    elif "0.1.0" not in notes.read_text(encoding="utf-8"):
        errors.append(f"{RELEASE_NOTES} does not reference version 0.1.0.")
    return errors


def banned_claim_errors() -> list[str]:
    errors: list[str] = []
    for path in tracked_text_files():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            lowered = line.lower()
            for phrase in BANNED_CLAIMS:
                if phrase in lowered:
                    errors.append(
                        f"{path.relative_to(ROOT)}:{line_number}: unsupported claim '{phrase}'"
                    )
    return errors


def collect(check_worktree: bool) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {
        "required release and approval files": required_file_errors(),
        "version consistency": version_errors(),
        "repository health": health_errors(),
        "high-confidence secrets": findings(),
        "local markdown links": broken_links(),
        "canonical repository URLs": non_canonical_references(),
        "unsupported public claims": banned_claim_errors(),
    }
    validate()  # raises on malformed YAML/JSON/JSONL/CFF
    if check_worktree:
        results["clean working tree"] = clean_worktree_errors()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only release preflight.")
    parser.add_argument(
        "--check", action="store_true", help="Run all checks (read-only; the only mode)."
    )
    parser.add_argument(
        "--allow-dirty", action="store_true", help="Skip the clean-working-tree check."
    )
    arguments = parser.parse_args()
    _ = arguments.check
    results = collect(check_worktree=not arguments.allow_dirty)
    total = 0
    for name, problems in results.items():
        if problems:
            total += len(problems)
            print(f"FAIL: {name}")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print(f"OK:   {name}")
    if total:
        print(f"\nRelease preflight failed with {total} issue(s).")
        return 1
    print("\nRelease preflight passed. Human approval is still required to tag or publish.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
