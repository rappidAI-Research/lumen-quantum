import json
import sys
from pathlib import Path

import pytest

from scripts.echelon_unattended import run_guarded

pytestmark = pytest.mark.unit


def test_unattended_guard_persists_logs_and_completion(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    log = tmp_path / "run.log"

    result = run_guarded(
        [sys.executable, "-c", "print('hello-echelON')"],
        run_id="run-ok",
        status_path=status,
        log_path=log,
        max_seconds=30,
        poll_seconds=0.01,
    )

    assert result == 0
    payload = json.loads(status.read_text(encoding="utf-8"))
    assert payload["state"] == "COMPLETED"
    assert "hello-echelON" in log.read_text(encoding="utf-8")


def test_unattended_guard_enforces_wall_time(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    log = tmp_path / "run.log"

    result = run_guarded(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        run_id="run-timeout",
        status_path=status,
        log_path=log,
        max_seconds=1,
        poll_seconds=0.01,
        terminate_timeout_seconds=1,
    )

    assert result == 124
    payload = json.loads(status.read_text(encoding="utf-8"))
    assert payload["state"] == "FAILED"
    assert payload["message"] == "wall-time limit exceeded"
