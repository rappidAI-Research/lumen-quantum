"""Check repository-relative Markdown links without making network requests."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "#")


def markdown_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / item for item in result.stdout.splitlines() if item]


def local_target(raw_target: str) -> str | None:
    target = raw_target.strip().strip("<>")
    if not target or target.startswith(EXTERNAL_PREFIXES):
        return None
    target = target.split(maxsplit=1)[0]
    return target.split("#", maxsplit=1)[0] or None


def broken_links() -> list[str]:
    broken: list[str] = []
    for document in markdown_files():
        for line_number, line in enumerate(
            document.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for match in LINK.finditer(line):
                target = local_target(match.group(1))
                if target is None:
                    continue
                resolved = (document.parent / target).resolve()
                if ROOT not in resolved.parents and resolved != ROOT:
                    broken.append(
                        f"{document.relative_to(ROOT)}:{line_number}: target escapes repository: {target}"
                    )
                elif not resolved.exists():
                    broken.append(
                        f"{document.relative_to(ROOT)}:{line_number}: missing target: {target}"
                    )
    return broken


def main() -> int:
    broken = broken_links()
    if broken:
        for item in broken:
            print(f"ERROR: {item}")
        return 1
    print("Local Markdown links passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
