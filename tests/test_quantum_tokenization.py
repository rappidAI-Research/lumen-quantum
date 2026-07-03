import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts.tokenize_quantum_data import tokenize_quantum_data
from scripts.train_quantum_tokenizer import train_quantum_tokenizer


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _text(index: int) -> str:
    return (
        f"Dokument {index}: Äpfel, Öl, Grüße und Fußgänger stehen in einem deutschen Testsatz. "
        "Lumen tokenisiert reproduzierbar und trennt Dokumente mit einem EOS-Token. "
        "Diese Zeile enthält genug wiederholbare Wörter für den lokalen Pilot-Tokenizer."
    )


def _tokenizer_config(tmp_path: Path, train_file: Path, tokenizer_dir: Path) -> Path:
    config = {
        "project": {"name": "Lumen Quantum", "tokenizer_name": "quantum-1-pilot-test"},
        "seed": 123,
        "data": {"train_file": str(train_file), "text_field": "text"},
        "tokenizer": {
            "type": "sentencepiece_bpe",
            "output_dir": str(tokenizer_dir),
            "vocab_size": 160,
            "character_coverage": 1.0,
            "hard_vocab_limit": True,
            "byte_fallback": False,
            "split_digits": True,
            "allow_whitespace_only_pieces": True,
            "remove_extra_whitespaces": False,
            "normalization_rule_name": "nfkc",
            "special_tokens": {
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"],
            },
        },
        "future_llama_config": {
            "architecture": "LlamaForCausalLM",
            "vocab_size": 160,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 3,
            "unk_token_id": 0,
            "tokenizer_class": "LlamaTokenizer",
        },
    }
    path = tmp_path / "tokenizer_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _pipeline_config(tmp_path: Path, tokenizer_dir: Path, cleaned_dir: Path, output_dir: Path, train_limit: int = 2000) -> Path:
    config = {
        "project": {"name": "Lumen Quantum", "dataset_name": "quantum-1-pilot-tokenized-test"},
        "seed": 123,
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
            "dir": str(output_dir),
            "manifest_file": "tokenization_manifest.json",
            "overwrite": False,
        },
        "packing": {
            "context_length": 512,
            "add_eos_after_each_document": True,
            "pad_remainder": True,
            "drop_empty_documents": True,
        },
        "limits": {
            "train_max_tokens": train_limit,
            "validation_max_tokens": 2000,
            "test_max_tokens": 2000,
        },
    }
    path = tmp_path / "quantum_1_pilot_data.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture()
def prepared_pipeline(tmp_path):
    cleaned_dir = tmp_path / "data" / "quantum" / "cleaned"
    tokenizer_dir = tmp_path / "tokenizer" / "quantum-1-pilot"
    output_dir = tmp_path / "data" / "quantum" / "tokenized" / "pilot"
    train_records = [{"id": f"train-{index}", "sha256": f"train-hash-{index}", "text": _text(index)} for index in range(10)]
    validation_records = [
        {"id": f"validation-{index}", "sha256": f"validation-hash-{index}", "text": _text(index + 20)}
        for index in range(3)
    ]
    test_records = [{"id": f"test-{index}", "sha256": f"test-hash-{index}", "text": _text(index + 40)} for index in range(3)]
    _write_jsonl(cleaned_dir / "train.jsonl", train_records)
    _write_jsonl(cleaned_dir / "validation.jsonl", validation_records)
    _write_jsonl(cleaned_dir / "test.jsonl", test_records)

    train_quantum_tokenizer(_tokenizer_config(tmp_path, cleaned_dir / "train.jsonl", tokenizer_dir))
    config_path = _pipeline_config(tmp_path, tokenizer_dir, cleaned_dir, output_dir)
    return config_path, output_dir, tokenizer_dir


def _load_split(output_dir: Path, split: str) -> dict:
    return torch.load(output_dir / f"{split}.pt", map_location="cpu", weights_only=True)


def test_tokenized_sequences_are_fixed_length_and_in_vocab(prepared_pipeline):
    config_path, output_dir, _tokenizer_dir = prepared_pipeline
    manifest_path = tokenize_quantum_data(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vocab_size = manifest["tokenizer"]["vocab_size"]

    for split in ["train", "validation", "test"]:
        data = _load_split(output_dir, split)
        assert data["input_ids"].ndim == 2
        assert data["input_ids"].shape[1] == 512
        assert data["attention_mask"].shape == data["input_ids"].shape
        assert data["labels"].shape == data["input_ids"].shape
        assert int(data["input_ids"].min()) >= 0
        assert int(data["input_ids"].max()) < vocab_size


def test_eos_is_used_after_documents(prepared_pipeline):
    config_path, output_dir, _tokenizer_dir = prepared_pipeline
    tokenize_quantum_data(config_path)
    data = _load_split(output_dir, "train")
    actual_tokens = data["input_ids"][data["attention_mask"].bool()].tolist()

    assert 2 in actual_tokens
    assert actual_tokens.count(2) == data["metadata"]["documents_used"]


def test_no_split_overlap_is_rejected(tmp_path):
    cleaned_dir = tmp_path / "data" / "quantum" / "cleaned"
    tokenizer_dir = tmp_path / "tokenizer" / "quantum-1-pilot"
    output_dir = tmp_path / "data" / "quantum" / "tokenized" / "pilot"
    _write_jsonl(cleaned_dir / "train.jsonl", [{"sha256": "same-hash", "text": _text(1)}])
    _write_jsonl(cleaned_dir / "validation.jsonl", [{"sha256": "same-hash", "text": _text(2)}])
    _write_jsonl(cleaned_dir / "test.jsonl", [{"sha256": "other-hash", "text": _text(3)}])
    train_quantum_tokenizer(_tokenizer_config(tmp_path, cleaned_dir / "train.jsonl", tokenizer_dir))
    config_path = _pipeline_config(tmp_path, tokenizer_dir, cleaned_dir, output_dir)

    with pytest.raises(ValueError, match="Split-Ueberschneidung"):
        tokenize_quantum_data(config_path)


def test_reproducible_tokenization_and_no_overwrite(prepared_pipeline):
    config_path, output_dir, _tokenizer_dir = prepared_pipeline
    tokenize_quantum_data(config_path)
    first = _load_split(output_dir, "train")

    with pytest.raises(FileExistsError):
        tokenize_quantum_data(config_path)

    tokenize_quantum_data(config_path, overwrite=True)
    second = _load_split(output_dir, "train")

    assert torch.equal(first["input_ids"], second["input_ids"])
    assert torch.equal(first["attention_mask"], second["attention_mask"])
    assert torch.equal(first["labels"], second["labels"])


def test_pilot_limit_works(prepared_pipeline):
    config_path, output_dir, _tokenizer_dir = prepared_pipeline
    manifest_path = tokenize_quantum_data(config_path, max_tokens=30)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_data = _load_split(output_dir, "train")

    assert manifest["splits"]["train"]["tokens_before_padding"] <= 30
    assert train_data["input_ids"].shape[1] == 512
    assert int(train_data["attention_mask"].sum().item()) <= 30
