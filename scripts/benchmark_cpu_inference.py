"""Local, reproducible CPU inference benchmark for a maintainer-supplied model.

Measures latency and throughput for a model directory the user provides
explicitly. It never downloads a model and never contacts the network. A
``--dry-run`` mode exercises the full measurement path with a dependency-free
mock generator so CI can test the tool without a model artifact.

No benchmark numbers are claimed in the repository until this tool is run against
a real local artifact and the record is reviewed by the maintainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = ROOT / "data" / "evals" / "cpu_benchmark_prompts.jsonl"
RECORD_SCHEMA = "cpu-inference-benchmark/1.1.0"
_UTC = timezone.utc  # noqa: UP017 (Python 3.10 runtime fallback; project targets 3.11+)


class Generator(Protocol):
    name: str

    def generate(self, prompt: str, max_new_tokens: int) -> int: ...


class MockGenerator:
    """Deterministic, dependency-free generator for the dry-run path."""

    name = "mock"

    def generate(self, prompt: str, max_new_tokens: int) -> int:
        checksum = 0
        for index in range(max_new_tokens):
            checksum = (checksum + len(prompt) + index) % 7
        return max_new_tokens


class TransformersGenerator:
    """Load a local Hugging Face model directory and run greedy generation."""

    name = "transformers"

    def __init__(self, model_path: Path, threads: int | None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int) -> int:
        torch = self._torch
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = int(output.shape[-1] - inputs["input_ids"].shape[-1])
        return max(generated, 0)


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        prompt = record.get("prompt") if isinstance(record, dict) else None
        if isinstance(prompt, str) and prompt:
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def run_benchmark(
    generator: Generator,
    prompts: list[str],
    max_new_tokens: int,
    warmup_runs: int,
    measurement_runs: int,
) -> dict[str, Any]:
    for _ in range(warmup_runs):
        for prompt in prompts:
            generator.generate(prompt, max_new_tokens)
    latencies_ms: list[float] = []
    per_second: list[float] = []
    total_tokens = 0
    for _ in range(measurement_runs):
        start = time.perf_counter()
        tokens = 0
        for prompt in prompts:
            tokens += generator.generate(prompt, max_new_tokens)
        elapsed = time.perf_counter() - start
        latencies_ms.append(elapsed * 1000.0)
        per_second.append(tokens / elapsed if elapsed > 0 else 0.0)
        total_tokens += tokens
    return {
        "generator": generator.name,
        "prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "warmup_runs": warmup_runs,
        "measurement_runs": measurement_runs,
        "total_generated_tokens": total_tokens,
        "latency_ms_per_run": latencies_ms,
        "latency_ms_mean": statistics.fmean(latencies_ms),
        "latency_ms_median": statistics.median(latencies_ms),
        "tokens_per_second_mean": statistics.fmean(per_second),
        "tokens_per_second_median": statistics.median(per_second),
    }


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_existing(base: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def _model_checksum(model_path: Path | None) -> str | None:
    if model_path is None:
        return None
    if model_path.is_file():
        return sha256_file(model_path)
    target = _first_existing(model_path, ("config.json",))
    return sha256_file(target) if target else None


def _tokenizer_checksum(model_path: Path | None) -> str | None:
    if model_path is None or not model_path.is_dir():
        return None
    target = _first_existing(model_path, ("tokenizer.model", "tokenizer.json"))
    return sha256_file(target) if target else None


def _git(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "tokenizers", "sentencepiece"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _total_ram_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def _peak_memory() -> dict[str, Any]:
    try:
        import resource
    except ImportError:
        return {"ru_maxrss": None, "unit": None, "note": "resource module unavailable here."}
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    unit = "bytes" if sys.platform == "darwin" else "kilobytes"
    return {"ru_maxrss": usage, "unit": unit, "note": "Process peak RSS; platform-dependent units."}


def build_record(
    metrics: dict[str, Any],
    model_path: Path | None,
    prompts_path: Path,
    max_new_tokens: int,
    threads: int | None,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "schema": RECORD_SCHEMA,
        "timestamp": datetime.now(_UTC).isoformat(),
        "dry_run": dry_run,
        "model_path": str(model_path) if model_path else None,
        "model_checksum": _model_checksum(model_path),
        "tokenizer_checksum": _tokenizer_checksum(model_path),
        "code_commit": _git("rev-parse", "HEAD"),
        "code_dirty": bool(_git("status", "--porcelain")),
        "environment": {
            "os": platform.system() or None,
            "platform": platform.platform(),
            "cpu": platform.processor() or None,
            "cpu_count": os.cpu_count(),
            "ram_bytes": _total_ram_bytes(),
            "threads_requested": threads,
            "python": platform.python_version(),
            "libraries": _library_versions(),
        },
        "prompt_file": str(prompts_path),
        "prompt_file_checksum": sha256_file(prompts_path),
        "generation": {"max_new_tokens": max_new_tokens, "do_sample": False},
        "peak_memory": _peak_memory(),
        "metrics": metrics,
        "limitations": [
            "Latency and throughput depend on hardware, thread count, and build flags.",
            "Dry-run mode uses a mock generator and does not reflect real model speed.",
            "Small synthetic prompts are not a task benchmark; results are operational only.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local CPU inference benchmark (no downloads).")
    parser.add_argument(
        "--model-path", type=Path, help="Local model directory (required unless --dry-run)."
    )
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measurement-runs", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with an in-process mock generator; no model or dependencies needed.",
    )
    arguments = parser.parse_args(argv)

    prompts = load_prompts(arguments.prompts)
    if arguments.dry_run:
        generator: Generator = MockGenerator()
        model_path: Path | None = None
    else:
        if arguments.model_path is None:
            parser.error("--model-path is required unless --dry-run is given")
        if not arguments.model_path.exists():
            parser.error(f"model path does not exist: {arguments.model_path}")
        generator = TransformersGenerator(arguments.model_path, arguments.threads)
        model_path = arguments.model_path

    metrics = run_benchmark(
        generator,
        prompts,
        arguments.max_new_tokens,
        arguments.warmup_runs,
        arguments.measurement_runs,
    )
    record = build_record(
        metrics,
        model_path,
        arguments.prompts,
        arguments.max_new_tokens,
        arguments.threads,
        bool(arguments.dry_run),
    )
    text = json.dumps(record, indent=2, ensure_ascii=False)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote benchmark record to {arguments.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
