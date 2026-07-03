import json
import sys
from pathlib import Path

from scripts.export_gguf import export_gguf, prepare_staging_model, validate_model_dir
from scripts.train_tokenizer import DEFAULT_SPECIAL_TOKENS, train_tokenizer


def _write_real_hf_model(root: Path) -> None:
    raw_dir = root.parent / "raw"
    tok_dir = root / "tokenizer"
    raw_dir.mkdir(parents=True)
    raw_text = (
        "Lumen ist ein lokaler Assistent.\n"
        "Äpfel, Öl und Grüße aus Köln.\n"
        "Das kleine Smoke-Modell prueft nur die Pipeline.\n"
    )
    (raw_dir / "text.txt").write_text(raw_text, encoding="utf-8")
    tokenizer = train_tokenizer(
        input_dir=raw_dir,
        output_dir=tok_dir,
        vocab_size=128,
        min_frequency=1,
        seed=123,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
        model_max_length=64,
        byte_fallback=False,
    )

    root.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": len(tokenizer),
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (root / "generation_config.json").write_text(
        json.dumps(
            {
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"not-a-real-model-for-wrapper-tests")


def _write_fake_llama_cpp(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text(
        """
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("--outfile", required=True)
parser.add_argument("--outtype", required=True)
args = parser.parse_args()

model_dir = Path(args.model_dir)
assert (model_dir / "config.json").exists()
assert (model_dir / "model.safetensors").exists()
assert (model_dir / "tokenizer.model").exists()
assert (model_dir / "tokenizer_config.json").exists()
assert (model_dir / "special_tokens_map.json").exists()
Path(args.outfile).write_bytes(b"GGUF")
""".strip(),
        encoding="utf-8",
    )


def test_prepare_staging_model_places_llama_sentencepiece_tokenizer_next_to_config(tmp_path):
    model_dir = tmp_path / "final"
    staging_dir = tmp_path / "staging"
    _write_real_hf_model(model_dir)

    validate_model_dir(model_dir)
    prepare_staging_model(model_dir, staging_dir)

    assert (staging_dir / "config.json").exists()
    assert (staging_dir / "model.safetensors").exists()
    assert (staging_dir / "tokenizer.model").exists()
    assert (staging_dir / "tokenizer_config.json").exists()
    assert (staging_dir / "special_tokens_map.json").exists()


def test_export_gguf_runs_llama_cpp_converter_wrapper(tmp_path):
    model_dir = tmp_path / "final"
    llama_cpp_dir = tmp_path / "llama.cpp"
    output_file = tmp_path / "models" / "smoke" / "quantum-smoke-f16.gguf"
    _write_real_hf_model(model_dir)
    _write_fake_llama_cpp(llama_cpp_dir)

    export_gguf(
        model_dir=model_dir,
        output_file=output_file,
        llama_cpp_dir=llama_cpp_dir,
        python_executable=sys.executable,
    )

    assert output_file.read_bytes() == b"GGUF"
