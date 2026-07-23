import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_parameter_preflight(tmp_path: Path) -> None:
    report = tmp_path / "preflight.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/echelon_preflight.py",
            "--report",
            str(report),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr

    data = json.loads(report.read_text(encoding="utf-8"))

    assert data["total_parameters"] == 506_333_440
    assert data["trainable_parameters"] == 506_333_440
    assert data["in_target_range"] is True
