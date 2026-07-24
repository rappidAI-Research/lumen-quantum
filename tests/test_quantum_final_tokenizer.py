import json
from pathlib import Path

import pytest
import yaml

from scripts.train_quantum_tokenizer import train_quantum_tokenizer
from scripts.validate_final_tokenizer import validate_final_tokenizer


def _texts() -> list[str]:
    base = [
        "Äpfel, Öl und Grüße aus Köln. Lumen trainiert den finalen deutschen Tokenizer.",
        "Fußgänger überqueren die Straße, während ein kleines Sprachmodell Wörter zerlegt.",
        "Schöne Grüße aus München: ä, ö, ü und ß müssen ohne UNK funktionieren.",
        "Der finale Tokenizer nutzt nur Trainingsdaten und keine Validation- oder Testdaten.",
        "Ein SentencePiece-BPE-Modell bleibt mit LlamaTokenizer und GGUF kompatibel.",
    ]
    return base * 50


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts),
        encoding="utf-8",
    )


def _tokenizer_config(tmp_path: Path, vocab_size: int = 160) -> tuple[Path, Path, Path]:
    train_file = tmp_path / "data" / "quantum" / "final" / "cleaned" / "train.jsonl"
    tokenizer_dir = tmp_path / "tokenizer" / "quantum-1"
    config = {
        "project": {"name": "rappidAI Quantum", "tokenizer_name": "quantum-1"},
        "seed": 20260704,
        "data": {"train_file": str(train_file), "text_field": "text"},
        "tokenizer": {
            "type": "sentencepiece_bpe",
            "output_dir": str(tokenizer_dir),
            "dir": str(tokenizer_dir),
            "model_file": "tokenizer.model",
            "manifest_file": "tokenizer_manifest.json",
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
            "max_position_embeddings": 512,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 3,
            "unk_token_id": 0,
            "tokenizer_class": "LlamaTokenizer",
        },
        "validation": {
            "normal_sentences": [
                "Äpfel, Öl und Grüße aus Köln.",
                "Fußgänger überqueren die Straße.",
            ],
            "roundtrip_sentences": ["Lumen trainiert den finalen deutschen Tokenizer."],
        },
    }
    path = tmp_path / "quantum_1_final_tokenizer.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    _write_jsonl(train_file, _texts())
    return path, train_file, tokenizer_dir


def _train_config(tmp_path: Path, tokenizer_dir: Path, vocab_size: int | str = 160) -> Path:
    config = {
        "project": {"name": "rappidAI Quantum", "model_name": "quantum-1-base"},
        "seed": 20260704,
        "tokenizer": {"dir": str(tokenizer_dir), "manifest_file": "tokenizer_manifest.json"},
        "model": {"vocab_size": vocab_size},
        "training": {"output_dir": str(tmp_path / "models" / "quantum-1-base")},
    }
    path = tmp_path / f"train_config_{vocab_size}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_final_tokenizer_validation_freezes_gguf_ready_tokenizer(tmp_path):
    config_path, train_file, tokenizer_dir = _tokenizer_config(tmp_path)
    train_quantum_tokenizer(config_path)
    train_config = _train_config(tmp_path, tokenizer_dir)

    report = validate_final_tokenizer(config_path, train_config, freeze=True)

    assert report["final_validation"]["gguf_ready"] is True
    assert (tokenizer_dir / "FINAL_FROZEN").exists()
    assert (tokenizer_dir / "freeze_manifest.json").exists()
    for filename in [
        "tokenizer.model",
        "tokenizer.vocab",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer_manifest.json",
    ]:
        assert filename in report["gguf_files"]
    assert report["token_ids"] == {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<pad>": 3,
        "<|system|>": 4,
        "<|user|>": 5,
        "<|assistant|>": 6,
    }
    manifest = json.loads((tokenizer_dir / "tokenizer_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_file"] == str(train_file)


def test_final_tokenizer_freeze_prevents_retraining(tmp_path):
    config_path, _train_file, tokenizer_dir = _tokenizer_config(tmp_path)
    train_quantum_tokenizer(config_path)
    validate_final_tokenizer(config_path, _train_config(tmp_path, tokenizer_dir), freeze=True)

    with pytest.raises(FileExistsError, match="eingefroren"):
        train_quantum_tokenizer(config_path)


def test_final_tokenizer_rejects_model_vocab_mismatch(tmp_path):
    config_path, _train_file, tokenizer_dir = _tokenizer_config(tmp_path)
    train_quantum_tokenizer(config_path)
    bad_train_config = _train_config(tmp_path, tokenizer_dir, vocab_size=159)

    with pytest.raises(ValueError, match="vocab_size"):
        validate_final_tokenizer(config_path, bad_train_config, freeze=False)


def test_final_tokenizer_rejects_validation_training_file(tmp_path):
    config_path, train_file, _tokenizer_dir = _tokenizer_config(tmp_path)
    validation_file = train_file.with_name("validation.jsonl")
    validation_file.write_text(train_file.read_text(encoding="utf-8"), encoding="utf-8")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["train_file"] = str(validation_file)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="train.jsonl"):
        train_quantum_tokenizer(config_path)
