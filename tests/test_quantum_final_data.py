import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts.validate_final_data import validate_final_data


def _text(split: str, index: int) -> str:
    return (
        f"Dies ist ein deutscher finaler {split} Text mit Index {index}. "
        "Er enthaelt genug Woerter fuer die Validierung, beschreibt Datenqualitaet, "
        "saubere Splits, reproduzierbare Hashes und robuste Tokenisierung. "
        "Lumen Quantum nutzt diese lokalen Tests nur fuer die finale Pipeline."
    )


def _record(split: str, index: int) -> dict:
    text = _text(split, index)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "id": f"{split}-{index}",
        "text": text,
        "sha256": digest,
        "split": split,
        "char_count": len(text),
        "word_count": len(text.split()),
        "approx_token_count": max(1, round(len(text) / 4)),
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _config(tmp_path: Path, use_pilot_paths: bool = False) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "data" / "quantum" / ("raw" if use_pilot_paths else "final")
    if use_pilot_paths:
        raw_dir = root
        cleaned_dir = tmp_path / "data" / "quantum" / "cleaned"
        manifest_dir = tmp_path / "data" / "quantum" / "manifests"
        report_dir = tmp_path / "data" / "quantum" / "reports"
    else:
        raw_dir = root / "raw"
        cleaned_dir = root / "cleaned"
        manifest_dir = root / "manifests"
        report_dir = root / "reports"
    tokenized_dir = tmp_path / "data" / "quantum" / "final" / "tokenized"
    tokenizer_dir = tmp_path / "tokenizer" / "quantum-1"

    data_config = {
        "project": {
            "name": "rappidAI Quantum",
            "dataset_name": "final-test",
            "description": "test",
        },
        "seed": 20260704,
        "source": {
            "hf_dataset": "epfml/FineWeb2-HQ",
            "hf_subset": "deu_Latn",
            "split": "train",
            "revision": "main",
            "license": "test",
            "citation": "test",
        },
        "targets": {
            "context_length": 512,
            "train_tokens": 1024,
            "validation_tokens": 512,
            "test_tokens": 512,
        },
        "paths": {
            "raw_dir": str(raw_dir),
            "cleaned_dir": str(cleaned_dir),
            "manifest_dir": str(manifest_dir),
            "report_dir": str(report_dir),
        },
        "download": {
            "streaming": True,
            "max_documents": 10,
            "max_raw_bytes": 100000,
            "shuffle_buffer_size": 10,
        },
        "cleaning": {
            "min_chars": 80,
            "max_chars": 1000,
            "min_words": 12,
            "max_repeated_char_run": 8,
            "max_non_letter_ratio": 0.40,
            "min_german_stopword_hits": 0,
            "normalize_whitespace": True,
            "remove_exact_duplicates": True,
            "boilerplate_patterns": [],
        },
        "sampling": {
            "split_seed": 20260704,
            "train_ratio": 0.98,
            "validation_ratio": 0.01,
            "test_ratio": 0.01,
        },
        "manifest": {"version": "quantum-1-final-test-v1", "notes": "test"},
    }
    tokenization_config = {
        "project": {"name": "rappidAI Quantum", "dataset_name": "final-tokenized-test"},
        "seed": 20260704,
        "tokenizer": {
            "dir": str(tokenizer_dir),
            "model_file": "tokenizer.model",
            "manifest_file": "tokenizer_manifest.json",
        },
        "input": {
            "train_file": str(cleaned_dir / "train.jsonl"),
            "validation_file": str(cleaned_dir / "validation.jsonl"),
            "test_file": str(cleaned_dir / "test.jsonl"),
            "text_field": "text",
            "document_hash_field": "sha256",
        },
        "output": {
            "dir": str(tokenized_dir),
            "manifest_file": "tokenization_manifest.json",
            "overwrite": False,
        },
        "packing": {
            "context_length": 512,
            "add_eos_after_each_document": True,
            "pad_remainder": True,
        },
        "limits": {
            "train_max_tokens": 4096,
            "validation_max_tokens": 2048,
            "test_max_tokens": 2048,
        },
    }

    data_config_path = tmp_path / "quantum_1_final_data.yaml"
    tokenization_config_path = tmp_path / "quantum_1_final_tokenizer.yaml"
    data_config_path.write_text(yaml.safe_dump(data_config, sort_keys=False), encoding="utf-8")
    tokenization_config_path.write_text(
        yaml.safe_dump(tokenization_config, sort_keys=False), encoding="utf-8"
    )
    return data_config_path, tokenization_config_path, cleaned_dir, tokenized_dir


def _write_manifest(data_config_path: Path) -> None:
    config = yaml.safe_load(data_config_path.read_text(encoding="utf-8"))
    manifest_dir = Path(config["paths"]["manifest_dir"])
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_name": config["project"]["dataset_name"],
        "dataset_version": config["manifest"]["version"],
        "source": config["source"],
        "download_date_utc": "2026-07-04T00:00:00+00:00",
        "license_hint": "test",
        "seed": config["seed"],
        "filter_rules": config["cleaning"],
        "document_counts": {"raw": 3, "cleaned": 3, "train": 1, "validation": 1, "test": 1},
        "text_amount": {
            "train": {"chars": 1, "words": 1},
            "validation": {"chars": 1, "words": 1},
            "test": {"chars": 1, "words": 1},
        },
        "estimated_tokens": {"train": 1, "validation": 1, "test": 1},
        "file_hashes": {},
    }
    (manifest_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _write_splits(cleaned_dir: Path, overlap: bool = False) -> dict[str, list[dict]]:
    records = {
        "train": [_record("train", 0), _record("train", 1)],
        "validation": [_record("validation", 0)],
        "test": [_record("test", 0)],
    }
    if overlap:
        records["validation"][0]["sha256"] = records["train"][0]["sha256"]
        records["validation"][0]["text"] = records["train"][0]["text"]
    for split, split_records in records.items():
        _write_jsonl(cleaned_dir / f"{split}.jsonl", split_records)
    return records


def _write_tokenized(
    tokenized_dir: Path,
    records: dict[str, list[dict]],
    vocab_size: int = 32,
    bad_token: bool = False,
    width: int = 512,
) -> None:
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"tokenizer": {"vocab_size": vocab_size}, "splits": {}}
    for split, split_records in records.items():
        input_ids = torch.arange(width, dtype=torch.long).unsqueeze(0) % vocab_size
        if bad_token and split == "train":
            input_ids[0, -1] = vocab_size
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        metadata = {
            "document_hashes": [record["sha256"] for record in split_records],
            "tokens_before_padding": int(attention_mask.sum().item()),
        }
        torch.save(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "metadata": metadata,
            },
            tokenized_dir / f"{split}.pt",
        )
        manifest["splits"][split] = {"tokens_before_padding": metadata["tokens_before_padding"]}
    (tokenized_dir / "tokenization_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def test_final_data_validation_accepts_disjoint_512_tokenized_splits(tmp_path):
    data_config_path, tokenization_config_path, cleaned_dir, tokenized_dir = _config(tmp_path)
    records = _write_splits(cleaned_dir)
    _write_manifest(data_config_path)
    _write_tokenized(tokenized_dir, records)

    report = validate_final_data(data_config_path, tokenization_config_path, require_tokenized=True)

    assert report["ok"] is True
    assert report["tokenized"]["splits"]["train"]["context_length"] == 512
    assert report["tokenized"]["splits"]["train"]["vocab_size"] == 32


def test_final_data_validation_rejects_split_overlap(tmp_path):
    data_config_path, tokenization_config_path, cleaned_dir, tokenized_dir = _config(tmp_path)
    records = _write_splits(cleaned_dir, overlap=True)
    _write_manifest(data_config_path)
    _write_tokenized(tokenized_dir, records)

    with pytest.raises(ValueError, match="Split-Ueberschneidung"):
        validate_final_data(data_config_path, tokenization_config_path, require_tokenized=True)


def test_final_data_validation_rejects_invalid_token_id(tmp_path):
    data_config_path, tokenization_config_path, cleaned_dir, tokenized_dir = _config(tmp_path)
    records = _write_splits(cleaned_dir)
    _write_manifest(data_config_path)
    _write_tokenized(tokenized_dir, records, bad_token=True)

    with pytest.raises(ValueError, match="Ungueltige Token-ID"):
        validate_final_data(data_config_path, tokenization_config_path, require_tokenized=True)


def test_final_data_validation_rejects_non_512_sequences(tmp_path):
    data_config_path, tokenization_config_path, cleaned_dir, tokenized_dir = _config(tmp_path)
    records = _write_splits(cleaned_dir)
    _write_manifest(data_config_path)
    _write_tokenized(tokenized_dir, records, width=511)

    with pytest.raises(ValueError, match="exakt 512"):
        validate_final_data(data_config_path, tokenization_config_path, require_tokenized=True)


def test_final_data_validation_rejects_pilot_paths(tmp_path):
    data_config_path, tokenization_config_path, cleaned_dir, tokenized_dir = _config(
        tmp_path, use_pilot_paths=True
    )
    records = _write_splits(cleaned_dir)
    _write_manifest(data_config_path)
    _write_tokenized(tokenized_dir, records)

    with pytest.raises(ValueError, match="data/quantum/final"):
        validate_final_data(data_config_path, tokenization_config_path, require_tokenized=True)
