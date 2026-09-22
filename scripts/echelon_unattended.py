"""Non-interactive process guard for supervised Quantum 1 Echelon jobs.

Run this wrapper under systemd (or an equivalent server-side supervisor).
It persists logs and atomic run status, forwards termination signals, enforces a
wall-time ceiling and never depends on the client SSH session.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from scripts.echelon_run_status import (
    initial_status,
    update_status,
    write_atomic,
)


def _terminate_process_group(process: subprocess.Popen[bytes], timeout: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=10)


def run_guarded(
    command: list[str],
    *,
    run_id: str,
    status_path: Path,
    log_path: Path,
    max_seconds: int,
    poll_seconds: float = 1.0,
    terminate_timeout_seconds: float = 120.0,
) -> int:
    if not command:
        raise ValueError("command must not be empty")
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    status = initial_status(run_id)
    status = update_status(status, message="supervisor starting")
    write_atomic(status_path, status)

    process: subprocess.Popen[bytes] | None = None
    received_signal: int | None = None

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signum

    previous_term = signal.signal(signal.SIGTERM, handle_signal)
    previous_int = signal.signal(signal.SIGINT, handle_signal)

    try:
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status = update_status(
                status,
                message=f"child started pid={process.pid}",
            )
            write_atomic(status_path, status)

            started = time.monotonic()
            while process.poll() is None:
                if received_signal is not None:
                    status = update_status(
                        status,
                        state="INTERRUPTED",
                        message=f"supervisor received signal {received_signal}",
                    )
                    write_atomic(status_path, status)
                    _terminate_process_group(process, terminate_timeout_seconds)
                    return 128 + received_signal

                if time.monotonic() - started >= max_seconds:
                    status = update_status(
                        status,
                        state="FAILED",
                        message="wall-time limit exceeded",
                    )
                    write_atomic(status_path, status)
                    _terminate_process_group(process, terminate_timeout_seconds)
                    return 124

                time.sleep(poll_seconds)

            return_code = int(process.returncode or 0)
            final_state = "COMPLETED" if return_code == 0 else "FAILED"
            status = update_status(
                status,
                state=final_state,
                message=f"child exited return_code={return_code}",
            )
            write_atomic(status_path, status)
            return return_code
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard an unattended Echelon command.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--status-path",
        type=Path,
        default=Path("logs/quantum-1-echelon/run-status.json"),
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/quantum-1-echelon/train.log"),
    )
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("provide the child command after --")

    return run_guarded(
        command,
        run_id=args.run_id,
        status_path=args.status_path,
        log_path=args.log_path,
        max_seconds=args.max_seconds,
    )


if __name__ == "__main__":
    sys.exit(main())
