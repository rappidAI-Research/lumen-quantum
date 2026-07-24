import json
from pathlib import Path

import torch
import yaml

from scripts.train_quantum_pilot import (
    TokenizedTensorDataset,
    detect_runtime_environment,
    train,
    validate_tokenized_data_dir,
)
from scripts.train_quantum_tokenizer import train_quantum_tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts),
        encoding="utf-8",
    )


def _tokenizer_config(
    tmp_path: Path, train_file: Path, tokenizer_dir: Path, vocab_size: int = 128
) -> Path:
    config = {
        "project": {"name": "rappidAI Quantum", "tokenizer_name": "cloud-pilot-test"},
        "seed": 123,
        "data": {"train_file": str(train_file), "text_field": "text"},
        "tokenizer": {
            "type": "sentencepiece_bpe",
            "output_dir": str(tokenizer_dir),
            "vocab_size": vocab_size,
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
            "vocab_size": vocab_size,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 3,
            "unk_token_id": 0,
            "tokenizer_class": "LlamaTokenizer",
        },
    }
    path = tmp_path / "tokenizer.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _write_tokenized_split(path: Path, vocab_size: int = 128, context_length: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    input_ids = torch.randint(4, vocab_size, (2, context_length), dtype=torch.long)
    input_ids[:, -1] = 2
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    torch.save(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "metadata": {
                "split": path.stem,
                "vocab_size": vocab_size,
                "context_length": context_length,
            },
        },
        path,
    )


def _tiny_cloud_config(tmp_path: Path, tokenizer_dir: Path, tokenized_dir: Path) -> Path:
    config = {
        "project": {
            "name": "rappidAI Quantum",
            "model_name": "quantum-1-base-test",
            "run_name": "dry-run",
        },
        "seed": 123,
        "tokenizer": {"dir": str(tokenizer_dir), "manifest_file": "tokenizer_manifest.json"},
        "data": {
            "tokenized_dir": str(tokenized_dir),
            "train_file": str(tokenized_dir / "train.pt"),
            "validation_file": str(tokenized_dir / "validation.pt"),
            "test_file": str(tokenized_dir / "test.pt"),
            "block_size": 16,
        },
        "model": {
            "architecture": "LlamaForCausalLM",
            "vocab_size": "auto",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 16,
            "rms_norm_eps": 0.000001,
            "rope_theta": 10000.0,
            "tie_word_embeddings": True,
            "attention_dropout": 0.0,
            "initializer_range": 0.02,
            "parameter_count_min": 1,
            "parameter_count_max": 1_000_000,
        },
        "training": {
            "output_dir": str(tmp_path / "models" / "cloud-pilot"),
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.0003,
            "weight_decay": 0.01,
            "warmup_steps": 1,
            "warmup_ratio": None,
            "lr_scheduler": "cosine",
            "max_steps": 1,
            "save_steps": 1,
            "eval_steps": 1,
            "logging_steps": 1,
            "max_grad_norm": 1.0,
            "num_workers": 0,
            "mixed_precision": "no",
            "resume_from_checkpoint": None,
        },
    }
    path = tmp_path / "cloud.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def test_cloud_pilot_yaml_has_expected_training_controls():
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "quantum_1_cloud_pilot.yaml").read_text(encoding="utf-8")
    )

    assert config["project"]["model_name"] == "quantum-1-base"
    assert config["data"]["tokenized_dir"] == "data/quantum/tokenized/pilot"
    assert config["data"]["block_size"] == 512
    assert config["training"]["max_steps"] == 100
    assert config["training"]["mixed_precision"] == "auto"
    assert config["training"]["batch_size"] > 0
    assert config["training"]["gradient_accumulation_steps"] > 0
    assert config["model"]["hidden_size"] == 512
    assert config["model"]["intermediate_size"] == 1536
    assert config["model"]["num_hidden_layers"] == 12


def test_runtime_detection_cpu_uses_no_mixed_precision(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime = detect_runtime_environment("auto")

    assert runtime["device"] == "cpu"
    assert runtime["mixed_precision"] == "no"
    assert runtime["cuda_available"] is False
    assert "torch_version" in runtime


def test_tokenized_dataset_validation_checks_shapes_and_vocab(tmp_path):
    tokenized_dir = tmp_path / "tokenized"
    for split in ["train", "validation", "test"]:
        _write_tokenized_split(tokenized_dir / f"{split}.pt", vocab_size=128, context_length=16)

    stats = validate_tokenized_data_dir(
        {"tokenized_dir": str(tokenized_dir)}, vocab_size=128, context_length=16
    )
    dataset = TokenizedTensorDataset(tokenized_dir / "train.pt", vocab_size=128, context_length=16)

    assert stats["train"]["sequences"] == 2
    assert len(dataset) == 2
    assert dataset[0]["input_ids"].shape == (16,)


def test_cloud_dry_run_builds_model_and_data_without_checkpoint(tmp_path):
    raw_train = tmp_path / "raw" / "train.jsonl"
    _write_jsonl(
        raw_train,
        [
            "Äpfel Öl Grüße Straße Lumen testet den Cloud Dry Run.",
            "Der Tokenizer bleibt lokal und das Modell startet zufällig.",
        ]
        * 20,
    )
    tokenizer_dir = tmp_path / "tokenizer" / "quantum-1-pilot"
    train_quantum_tokenizer(_tokenizer_config(tmp_path, raw_train, tokenizer_dir))

    tokenized_dir = tmp_path / "data" / "quantum" / "tokenized" / "pilot"
    for split in ["train", "validation", "test"]:
        _write_tokenized_split(tokenized_dir / f"{split}.pt", vocab_size=128, context_length=16)

    config_path = _tiny_cloud_config(tmp_path, tokenizer_dir, tokenized_dir)
    output_dir = train(config_path, dry_run=True)

    assert output_dir == tmp_path / "models" / "cloud-pilot"
    assert not output_dir.exists()
