"""Scan tracked text for a small set of high-confidence credential formats."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "private key": re.compile("-----BEGIN " + "(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile("gh" + r"[pousr]_[A-Za-z0-9]{30,}"),
    "OpenAI-style key": re.compile("sk" + r"-[A-Za-z0-9_-]{30,}"),
    "AWS access key": re.compile("AK" + r"IA[0-9A-Z]{16}"),
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / item for item in result.stdout.splitlines() if item]


def findings() -> list[str]:
    hits: list[str] = []
    for path in tracked_files():
        if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    hits.append(f"{path.relative_to(ROOT)}:{line_number}: {label}")
    return hits


def main() -> int:
    hits = findings()
    if hits:
        for hit in hits:
            print(f"ERROR: {hit}")
        return 1
    print("High-confidence secret-pattern scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
