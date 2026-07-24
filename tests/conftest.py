"""Central test classification for default CPU-only CI selection."""

from pathlib import Path

import pytest

INTEGRATION_FILES = {
    "test_checkpoint.py",
    "test_cloud_pilot_config.py",
    "test_export_gguf.py",
    "test_model.py",
    "test_quantum_final_data.py",
    "test_quantum_final_tokenizer.py",
    "test_quantum_model.py",
    "test_quantum_tokenization.py",
    "test_quantum_tokenizer.py",
    "test_tokenizer.py",
}
SLOW_FILES = {
    "test_cloud_pilot_config.py",
    "test_quantum_final_tokenizer.py",
    "test_quantum_model.py",
    "test_quantum_tokenizer.py",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        filename = Path(str(item.fspath)).name
        category = pytest.mark.integration if filename in INTEGRATION_FILES else pytest.mark.unit
        item.add_marker(category)
        if filename in SLOW_FILES:
            item.add_marker(pytest.mark.slow)
