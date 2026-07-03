import json
from pathlib import Path

import yaml

from scripts.build_data_manifest import run as build_manifest
from scripts.clean_quantum_data import run as clean_data
from scripts.download_quantum_data import run as download_data
from scripts.sample_quantum_data import run as sample_data


def _write_config(tmp_path: Path) -> Path:
    config = {
        "project": {
            "name": "Lumen Quantum",
            "dataset_name": "quantum-1-data-test",
            "description": "test",
        },
        "seed": 123,
        "paths": {
            "raw_dir": str(tmp_path / "data" / "quantum" / "raw"),
            "cleaned_dir": str(tmp_path / "data" / "quantum" / "cleaned"),
            "manifest_dir": str(tmp_path / "data" / "quantum" / "manifests"),
        },
        "download": {
            "write_seed_sample": True,
            "allow_network_default": False,
            "max_documents_per_source": 20,
            "sources": [],
        },
        "cleaning": {
            "min_chars": 80,
            "max_chars": 12000,
            "min_words": 12,
            "max_repeated_char_run": 6,
            "max_non_letter_ratio": 0.35,
            "min_german_stopword_hits": 2,
            "normalize_whitespace": True,
            "remove_exact_duplicates": True,
            "reject_patterns": ["cookie", "javascript"],
        },
        "sampling": {
            "max_documents": 20,
            "train_ratio": 0.8,
            "validation_ratio": 0.1,
            "test_ratio": 0.1,
        },
        "manifest": {
            "version": "test-v0",
            "notes": "test",
        },
    }
    config_path = tmp_path / "quantum_1_data.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_quantum_data_pipeline_is_reproducible_and_disjoint(tmp_path):
    config_path = _write_config(tmp_path)

    raw_file = download_data(config_path)
    cleaned_file = clean_data(config_path)
    split_paths = sample_data(config_path)
    manifest_json, manifest_md = build_manifest(config_path)

    assert raw_file.exists()
    assert cleaned_file.exists()
    assert split_paths["train"].exists()
    assert split_paths["validation"].exists()
    assert split_paths["test"].exists()
    assert manifest_json.exists()
    assert manifest_md.exists()

    train = _read_jsonl(split_paths["train"])
    validation = _read_jsonl(split_paths["validation"])
    test = _read_jsonl(split_paths["test"])
    train_hashes = {record["sha256"] for record in train}
    validation_hashes = {record["sha256"] for record in validation}
    test_hashes = {record["sha256"] for record in test}

    assert train_hashes
    assert validation_hashes
    assert test_hashes
    assert train_hashes.isdisjoint(validation_hashes)
    assert train_hashes.isdisjoint(test_hashes)
    assert validation_hashes.isdisjoint(test_hashes)

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["seed"] == 123
    assert manifest["files"]["train"]["documents"] == len(train)
