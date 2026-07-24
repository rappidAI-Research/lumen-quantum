from pathlib import Path

from scripts.check_markdown_links import broken_links
from scripts.check_secrets import findings
from scripts.repository_health import ROOT, health_errors, unregistered_gitlinks
from scripts.validate_repository_configs import validate


def test_no_gitlink_exists_without_submodule_metadata() -> None:
    assert unregistered_gitlinks() == []


def test_repository_health_contract() -> None:
    assert health_errors() == []


def test_tracked_configs_and_reports_parse() -> None:
    validated = validate()
    assert "pyproject.toml" not in validated
    assert "configs/echelon/quantum-1-echelon-base.yaml" in validated
    assert "data/evals/quantum_1_base_v1.jsonl" in validated


def test_local_markdown_links_resolve() -> None:
    assert broken_links() == []


def test_high_confidence_secret_patterns_are_absent() -> None:
    assert findings() == []


def test_repository_root_is_resolved_from_the_script() -> None:
    assert ROOT == Path(__file__).resolve().parents[1]
