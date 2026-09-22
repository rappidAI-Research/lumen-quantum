import json
import sys
from pathlib import Path

import pytest

from scripts.echelon_systemd_entrypoint import load_env_file, parse_child_command, run_from_env

pytestmark = pytest.mark.unit


def _env_file(tmp_path: Path, command: list[str]) -> Path:
    path = tmp_path / "run.env"
    path.write_text(
        "\n".join(
            [
                "ECHELON_RUN_ID=systemd-test",
                f"ECHELON_STATUS_PATH={tmp_path / 'status.json'}",
                f"ECHELON_LOG_PATH={tmp_path / 'run.log'}",
                "ECHELON_MAX_SECONDS=30",
                "ECHELON_CHILD_JSON=" + json.dumps(command),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_entrypoint_runs_without_shell(tmp_path: Path) -> None:
    env_file = _env_file(
        tmp_path,
        [sys.executable, "-c", "print('systemd-echelON-ok')"],
    )
    assert run_from_env(env_file) == 0
    assert "systemd-echelON-ok" in (tmp_path / "run.log").read_text(encoding="utf-8")


def test_env_file_rejects_unknown_keys(tmp_path: Path) -> None:
    env_file = _env_file(tmp_path, [sys.executable, "-c", "pass"])
    with env_file.open("a", encoding="utf-8") as handle:
        handle.write("AWS_SECRET_ACCESS_KEY=forbidden\n")

    with pytest.raises(ValueError, match="unexpected environment keys"):
        load_env_file(env_file)


def test_child_json_must_be_argv_array() -> None:
    with pytest.raises(ValueError, match="non-empty JSON array"):
        parse_child_command('"python train.py"')
