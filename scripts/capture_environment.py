"""Capture a resolved runtime environment snapshot as JSON.

Records Python, OS, platform, CPU, key ML library versions, every installed
package version, the Git commit, and the working-tree dirty status. CUDA and GPU
details are only included when PyTorch is importable. Nothing is guessed and no
platform-specific assumption is made.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KEY_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "sentencepiece",
    "tokenizers",
    "safetensors",
    "numpy",
    "PyYAML",
)


def _git(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            packages[name] = dist.version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def key_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in KEY_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_details() -> dict[str, Any]:
    details: dict[str, Any] = {
        "available": False,
        "version": None,
        "cuda_version": None,
        "cuda_available": None,
        "gpus": None,
    }
    try:
        import torch
    except ImportError:
        return details
    details["available"] = True
    details["version"] = getattr(torch, "__version__", None)
    details["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        cuda_available = bool(torch.cuda.is_available())
        details["cuda_available"] = cuda_available
        if cuda_available:
            details["gpus"] = [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
    except Exception as error:  # pragma: no cover - defensive; driver/runtime issues
        details["cuda_error"] = str(error)
    return details


def capture() -> dict[str, Any]:
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "os": {"system": platform.system() or None, "release": platform.release() or None},
        "platform": platform.platform(),
        "cpu": {
            "processor": platform.processor() or None,
            "machine": platform.machine() or None,
        },
        "torch": torch_details(),
        "key_libraries": key_versions(),
        "installed_packages": installed_packages(),
        "git": {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain"))},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a runtime environment snapshot as JSON.")
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout.")
    arguments = parser.parse_args()
    snapshot = capture()
    text = json.dumps(snapshot, indent=2, ensure_ascii=False)
    if arguments.output is not None:
        arguments.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote environment snapshot to {arguments.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
