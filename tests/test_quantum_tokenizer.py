import json
from pathlib import Path

import pytest
import yaml

from scripts.train_quantum_tokenizer import train_quantum_tokenizer
from scripts.validate_quantum_tokenizer import validate_quantum_tokenizer


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts),
        encoding="utf-8",
    )


def _texts() -> list[str]:
    base = [
        "Äpfel, Öl und Grüße aus Köln. Lumen trainiert einen eigenen deutschen Tokenizer.",
        "Fußgänger überqueren die Straße, während ein kleines Sprachmodell Wörter zerlegt.",
        "Schöne Grüße aus München: ä, ö, ü und ß müssen ohne UNK funktionieren.",
        "Der Pilot-Tokenizer nutzt nur Trainingsdaten und keine Validation- oder Testdaten.",
        "Ein SentencePiece-BPE-Modell bleibt mit LlamaTokenizer und GGUF kompatibel.",
    ]
    return base * 30


def _config(tmp_path: Path, train_file: Path, vocab_size: int = 128) -> Path:
    config = {
        "project": {"name": "rappidAI Quantum", "tokenizer_name": "quantum-1-pilot-test"},
        "seed": 123,
        "data": {"train_file": str(train_file), "text_field": "text"},
        "tokenizer": {
            "type": "sentencepiece_bpe",
            "output_dir": str(tmp_path / "tokenizer" / "quantum-1-pilot"),
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
        "validation": {
            "normal_sentences": [
                "Äpfel, Öl und Grüße aus Köln.",
                "Fußgänger überqueren die Straße.",
            ],
            "roundtrip_sentences": ["Lumen trainiert einen eigenen deutschen Tokenizer."],
        },
    }
    path = tmp_path / "quantum_1_tokenizer_pilot.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def test_train_and_validate_quantum_pilot_tokenizer(tmp_path):
    train_file = tmp_path / "data" / "quantum" / "cleaned" / "train.jsonl"
    _write_jsonl(train_file, _texts())
    config_path = _config(tmp_path, train_file)

    tokenizer_dir = train_quantum_tokenizer(config_path)
    report = validate_quantum_tokenizer(config_path)

    for filename in [
        "tokenizer.model",
        "tokenizer.vocab",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer_manifest.json",
    ]:
        assert (tokenizer_dir / filename).exists()

    manifest = json.loads((tokenizer_dir / "tokenizer_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_file"] == str(train_file)
    assert manifest["requested_vocab_size"] == 128
    assert manifest["actual_vocab_size"] == 128
    assert manifest["token_ids"]["<unk>"] == 0
    assert manifest["token_ids"]["<s>"] == 1
    assert manifest["token_ids"]["</s>"] == 2
    assert manifest["token_ids"]["<pad>"] == 3
    assert manifest["token_ids"]["<|system|>"] > 3
    assert manifest["token_ids"]["<|user|>"] > 3
    assert manifest["token_ids"]["<|assistant|>"] > 3
    assert report["actual_vocab_size"] == 128
    assert report["gguf_pre_export_checks"]["classic_llama_ids"] is True


def test_quantum_tokenizer_refuses_validation_or_test_file(tmp_path):
    validation_file = tmp_path / "data" / "quantum" / "cleaned" / "validation.jsonl"
    _write_jsonl(validation_file, _texts())
    config_path = _config(tmp_path, validation_file)

    with pytest.raises(ValueError, match="train.jsonl"):
        train_quantum_tokenizer(config_path)
