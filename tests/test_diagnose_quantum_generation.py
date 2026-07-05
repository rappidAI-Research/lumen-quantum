import json
from pathlib import Path

import pytest
from transformers import LlamaConfig

from scripts.diagnose_quantum_generation import (
    android_capture_template,
    build_generation_record,
    compare_tokenizer_files,
    ensure_gguf_file,
    resolve_output_dir,
    tokenizer_roundtrip_for_prompts,
    validate_android_capture_record,
    validate_vocab_alignment,
)


class StableFakeTokenizer:
    def encode(self, text, out_type=int):
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join(chr(token_id) for token_id in ids)


class UnstableFakeTokenizer:
    def encode(self, text, out_type=int):
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join(chr(token_id) for token_id in ids) + "!"


def _config(tmp_path: Path) -> dict:
    return {
        "project": {"model_name": "quantum-1.6-pilot"},
        "seed": 20260705,
        "paths": {
            "output_dir": str(tmp_path / "data" / "diagnostics" / "quantum-1.6-pilot"),
            "gguf_file": str(tmp_path / "exports" / "quantum-1.6-pilot-v1.6.0-f16.gguf"),
        },
        "expected": {"context_length": 512},
        "prompts": ["Berlin ist"],
        "generation": {
            "max_new_tokens": 64,
            "modes": {
                "sampling": {
                    "label": "controlled_sampling",
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                }
            },
        },
        "android": {
            "expected_model_id": "quantum-1.6-pilot-v1.6.0-f16",
            "expected_ui_model_id": "quantum-1.6-pilot-v1.6.0-f16",
            "require_same_gguf_sha256_as_terminal": True,
            "require_same_file_size_as_terminal": True,
            "require_ui_loaded_model_match": True,
            "required_capture_fields": [
                "model_id",
                "ui_active_model_id",
                "local_model_path",
                "file_size_bytes",
                "sha256",
                "prompt",
                "mode",
                "sampling",
                "raw_stream_fragments",
            ],
        },
    }


def test_tokenizer_roundtrip_reports_stable_ids():
    report = tokenizer_roundtrip_for_prompts(StableFakeTokenizer(), ["Berlin ist", "Nutzer: Hallo\nLumen:"])

    assert report["all_stable"] is True
    assert report["stable_count"] == 2
    assert report["results"][0]["first_ids"] == report["results"][0]["second_ids"]


def test_tokenizer_roundtrip_detects_unstable_ids():
    report = tokenizer_roundtrip_for_prompts(UnstableFakeTokenizer(), ["Berlin ist"])

    assert report["all_stable"] is False
    assert report["results"][0]["stable"] is False


def test_wrong_tokenizer_is_detected(tmp_path):
    model_dir = tmp_path / "models" / "quantum-1.6-pilot" / "final"
    tokenizer_dir = tmp_path / "tokenizer" / "quantum-1"
    model_dir.mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True)
    (model_dir / "tokenizer.model").write_bytes(b"model-tokenizer")
    (tokenizer_dir / "tokenizer.model").write_bytes(b"different-tokenizer")

    with pytest.raises(ValueError, match="Tokenizer mismatch"):
        compare_tokenizer_files(model_dir, tokenizer_dir)


def test_matching_tokenizer_reports_hashes(tmp_path):
    model_dir = tmp_path / "models" / "quantum-1.6-pilot" / "final"
    tokenizer_dir = tmp_path / "tokenizer" / "quantum-1"
    model_dir.mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True)
    payload = b"same-tokenizer"
    (model_dir / "tokenizer.model").write_bytes(payload)
    (tokenizer_dir / "tokenizer.model").write_bytes(payload)

    report = compare_tokenizer_files(model_dir, tokenizer_dir)

    assert report["byte_identical"] is True
    assert report["model_tokenizer_sha256"] == report["configured_tokenizer_sha256"]


def test_wrong_vocab_size_is_detected():
    model_config = LlamaConfig(vocab_size=128)

    with pytest.raises(ValueError, match="Vokabulargroesse"):
        validate_vocab_alignment(model_config, tokenizer_vocab_size=129, expected_vocab_size=128)


def test_expected_vocab_size_is_detected():
    model_config = LlamaConfig(vocab_size=128)

    with pytest.raises(ValueError, match="erwartet"):
        validate_vocab_alignment(model_config, tokenizer_vocab_size=128, expected_vocab_size=256)


def test_report_structure_for_generation_record():
    record = build_generation_record(
        prompt_index=0,
        prompt="Berlin ist",
        mode_name="greedy_deterministic",
        mode_config={"do_sample": False, "temperature": 0.0, "top_p": 1.0, "max_new_tokens": 64},
        prompt_token_ids=[10, 11],
        generation_input_ids=[1, 10, 11],
        output_token_ids=[1, 10, 11, 12, 2],
        generated_token_ids=[12, 2],
        decoded_text="Berlin ist groß",
        decoded_generated_text=" groß",
        unk_token_id=0,
        eos_token_id=2,
    )

    assert record["prompt"] == "Berlin ist"
    assert record["prompt_token_ids"] == [10, 11]
    assert record["generated_token_ids"] == [12, 2]
    assert record["unknown_token_count"] == 0
    assert record["eos_behavior"]["contains_eos"] is True
    assert "unusual_characters" in record


def test_diagnosis_paths_are_model_isolated(tmp_path):
    output_dir = resolve_output_dir(_config(tmp_path))

    assert output_dir.name == "quantum-1.6-pilot"
    assert output_dir.exists()


def test_diagnosis_path_without_model_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="modellisoliert"):
        resolve_output_dir(_config(tmp_path), tmp_path / "data" / "diagnostics" / "other-model")


def test_missing_gguf_file_has_clear_error(tmp_path):
    missing = tmp_path / "exports" / "missing.gguf"

    with pytest.raises(FileNotFoundError, match="GGUF-Datei nicht gefunden"):
        ensure_gguf_file(missing)


def test_android_capture_template_contains_required_logging_fields(tmp_path):
    template = android_capture_template(_config(tmp_path), gguf_sha256="abc", gguf_size=123)

    assert template["expected_gguf_sha256"] == "abc"
    assert template["expected_file_size_bytes"] == 123
    assert "raw_stream_fragments" in template["required_fields"]
    assert template["example_record"]["sampling"]["top_k"] == 40


def test_android_capture_detects_wrong_loaded_model_hash(tmp_path):
    config = _config(tmp_path)
    record = {
        "model_id": "quantum-1.6-pilot-v1.6.0-f16",
        "ui_active_model_id": "quantum-1.6-pilot-v1.6.0-f16",
        "local_model_path": "/data/user/0/app/files/model.gguf",
        "file_size_bytes": 123,
        "sha256": "wrong",
        "prompt": "Berlin ist",
        "mode": "controlled_sampling",
        "sampling": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "seed": 20260705,
            "max_tokens": 64,
            "context_length": 512,
        },
        "raw_stream_fragments": ["Berlin", " ist"],
    }

    validation = validate_android_capture_record(record, config, gguf_sha256="expected", gguf_size=123)

    assert validation["issue_count"] == 1
    assert validation["issues"][0].startswith("gguf_sha256_mismatch")


def test_minimal_report_json_can_be_written(tmp_path):
    report_path = tmp_path / "data" / "diagnostics" / "quantum-1.6-pilot" / "pytorch_generation_report.json"
    report_path.parent.mkdir(parents=True)
    payload = {
        "model_name": "quantum-1.6-pilot",
        "preflight": {"vocab_check": {"ok": True}},
        "generations": [
            build_generation_record(
                prompt_index=0,
                prompt="Ein Computer ist",
                mode_name="controlled_sampling",
                mode_config={"do_sample": True, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 64},
                prompt_token_ids=[4, 5],
                generation_input_ids=[1, 4, 5],
                output_token_ids=[1, 4, 5, 6],
                generated_token_ids=[6],
                decoded_text="Ein Computer ist ein Werkzeug",
                decoded_generated_text=" ein Werkzeug",
                unk_token_id=0,
                eos_token_id=2,
            )
        ],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["model_name"] == "quantum-1.6-pilot"
    assert loaded["generations"][0]["mode"] == "controlled_sampling"
