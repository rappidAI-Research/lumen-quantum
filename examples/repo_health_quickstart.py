"""CPU- and network-free quickstart: validate configs and check repository health.

Run from the repository root:

    python examples/repo_health_quickstart.py

It parses tracked configs, runs the repository-health, secret, Markdown-link and
canonical-URL checks, writes a tiny synthetic dataset to a temporary directory,
and prints a summary. It installs nothing, downloads nothing, and needs no GPU.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_LINES = (
    "Lumen ist ein lokaler Testassistent.",
    "Ein kleines Modell lernt zuerst nur den Ablauf.",
    "Reproduzierbarkeit bedeutet nachvollziehbare Schritte.",
)


def _ensure_repo_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def write_synthetic_dataset(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "train.txt"
    target.write_text("\n".join(SYNTHETIC_LINES) + "\n", encoding="utf-8")
    return target


def run() -> dict[str, list[str]]:
    _ensure_repo_on_path()
    from scripts.check_canonical_urls import non_canonical_references
    from scripts.check_markdown_links import broken_links
    from scripts.check_secrets import findings
    from scripts.repository_health import health_errors
    from scripts.validate_repository_configs import validate

    validate()
    return {
        "repository health": health_errors(),
        "high-confidence secrets": findings(),
        "local markdown links": broken_links(),
        "canonical repository URLs": non_canonical_references(),
    }


def main() -> int:
    results = run()
    with tempfile.TemporaryDirectory() as directory:
        dataset = write_synthetic_dataset(Path(directory) / "data")
        line_count = len(dataset.read_text(encoding="utf-8").splitlines())
    total = 0
    for name, problems in results.items():
        total += len(problems)
        status = "OK" if not problems else f"{len(problems)} issue(s)"
        print(f"{status:>12}  {name}")
        for problem in problems:
            print(f"              - {problem}")
    print(f"{'OK':>12}  synthetic dataset ({line_count} lines) written and removed")
    if total:
        print(f"\nQuickstart found {total} issue(s).")
        return 1
    print("\nQuickstart passed. No GPU, network, or downloads were used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
