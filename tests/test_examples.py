from pathlib import Path

from examples import minimal_ml_pipeline, repo_health_quickstart

EXAMPLES = Path(repo_health_quickstart.__file__).resolve().parent


def test_example_files_exist() -> None:
    assert (EXAMPLES / "repo_health_quickstart.py").is_file()
    assert (EXAMPLES / "minimal_ml_pipeline.py").is_file()
    assert (EXAMPLES / "README.md").is_file()


def test_repo_health_quickstart_reports_no_issues() -> None:
    results = repo_health_quickstart.run()
    for name, problems in results.items():
        assert problems == [], (name, problems)


def test_synthetic_dataset_is_written(tmp_path: Path) -> None:
    dataset = repo_health_quickstart.write_synthetic_dataset(tmp_path / "data")
    assert dataset.is_file()
    assert dataset.read_text(encoding="utf-8").strip()


def test_minimal_pipeline_dependency_probe_returns_bool() -> None:
    assert isinstance(minimal_ml_pipeline.dependencies_available(), bool)
