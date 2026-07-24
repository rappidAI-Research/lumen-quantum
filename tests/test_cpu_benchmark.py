from pathlib import Path

from scripts.benchmark_cpu_inference import (
    DEFAULT_PROMPTS,
    MockGenerator,
    load_prompts,
    main,
    run_benchmark,
)


def test_default_prompts_load() -> None:
    prompts = load_prompts(DEFAULT_PROMPTS)
    assert prompts
    assert all(isinstance(prompt, str) and prompt for prompt in prompts)


def test_run_benchmark_reports_metrics_without_dependencies() -> None:
    prompts = ["hello world", "another prompt"]
    metrics = run_benchmark(
        MockGenerator(), prompts, max_new_tokens=8, warmup_runs=1, measurement_runs=3
    )
    assert metrics["measurement_runs"] == 3
    assert metrics["total_generated_tokens"] == 2 * 3 * 8
    assert len(metrics["latency_ms_per_run"]) == 3
    assert metrics["tokens_per_second_mean"] >= 0.0


def test_dry_run_main_writes_a_record(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    exit_code = main(
        ["--dry-run", "--measurement-runs", "2", "--warmup-runs", "0", "--output", str(output)]
    )
    assert exit_code == 0
    assert output.exists()
