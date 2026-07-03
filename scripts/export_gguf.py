"""Exportiert das lokale Smoke-Modell nach GGUF mit llama.cpp.

Der Wrapper laedt keine Modellgewichte in Python und verwendet kein
from_pretrained. Er prueft nur die lokale Hugging-Face-Struktur, erstellt bei
Bedarf eine HF-kompatible Staging-Kopie mit Tokenizer-Dateien im Root und ruft
den llama.cpp-Konverter auf.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


LOGGER = logging.getLogger("lumen.export_gguf")

EXPECTED_LLAMA_SPECIAL_TOKENS = {
    "unk_token": ("<unk>", 0),
    "bos_token": ("<s>", 1),
    "eos_token": ("</s>", 2),
    "pad_token": ("<pad>", 3),
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def find_llama_converter(llama_cpp_dir: str | Path) -> Path:
    root = Path(llama_cpp_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"llama.cpp-Verzeichnis nicht gefunden: {root}. "
            "Gib den Pfad mit --llama-cpp-dir an."
        )

    candidates = [
        root / "convert_hf_to_gguf.py",
        root / "convert.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Kein llama.cpp-Konverter gefunden. Erwartet wurde "
        f"{root / 'convert_hf_to_gguf.py'} oder {root / 'convert.py'}."
    )


def find_weight_files(model_dir: str | Path) -> list[Path]:
    root = Path(model_dir)
    patterns = [
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(root.glob(pattern)))

    index_files = [
        root / "model.safetensors.index.json",
        root / "pytorch_model.bin.index.json",
    ]
    files.extend(path for path in index_files if path.exists())
    return sorted(set(files))


def find_tokenizer_dir(model_dir: str | Path) -> Path:
    root = Path(model_dir)
    if (root / "tokenizer.model").exists():
        return root
    nested = root / "tokenizer"
    if (nested / "tokenizer.model").exists():
        return nested
    raise FileNotFoundError(
        f"SentencePiece-Tokenizer nicht gefunden. Erwartet {root / 'tokenizer.model'} "
        f"oder {nested / 'tokenizer.model'}. Trainiere den Tokenizer mit scripts/train_tokenizer.py neu."
    )


def validate_model_dir(model_dir: str | Path) -> tuple[Path, list[Path], Path]:
    root = Path(model_dir)
    if not root.exists():
        raise FileNotFoundError(f"Modellordner nicht gefunden: {root}")

    config_file = root / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json fehlt: {config_file}")

    with config_file.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("model_type") != "llama":
        raise ValueError(
            f"config.json ist nicht als LLaMA-Modell markiert: model_type={config.get('model_type')!r}"
        )
    architectures = config.get("architectures") or []
    if architectures and "LlamaForCausalLM" not in architectures:
        raise ValueError(
            "config.json enthaelt keine LlamaForCausalLM-Architektur: "
            f"architectures={architectures!r}"
        )

    weight_files = find_weight_files(root)
    if not weight_files:
        raise FileNotFoundError(
            f"Keine Modellgewichte in {root} gefunden. Erwartet model.safetensors "
            "oder pytorch_model.bin."
        )

    tokenizer_dir = find_tokenizer_dir(root)
    validate_llama_sentencepiece_compatibility(root, config, tokenizer_dir)
    return config_file, weight_files, tokenizer_dir


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_llama_sentencepiece_compatibility(model_dir: Path, config: dict, tokenizer_dir: Path) -> None:
    try:
        import sentencepiece as spm
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "sentencepiece ist fuer die GGUF-Kompatibilitaetspruefung erforderlich. "
            "Installiere: pip install -r requirements.txt"
        ) from exc

    tokenizer_model = tokenizer_dir / "tokenizer.model"
    tokenizer_config = read_json_if_exists(tokenizer_dir / "tokenizer_config.json")
    special_tokens_map = read_json_if_exists(tokenizer_dir / "special_tokens_map.json")

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    piece_size = int(sp.get_piece_size())
    model_vocab_size = int(config.get("vocab_size", -1))
    if model_vocab_size != piece_size:
        raise ValueError(
            f"vocab_size passt nicht: config.json={model_vocab_size}, tokenizer.model={piece_size}."
        )

    if tokenizer_config.get("tokenizer_class") != "LlamaTokenizer":
        raise ValueError(
            "tokenizer_config.json muss tokenizer_class='LlamaTokenizer' enthalten, "
            f"gefunden: {tokenizer_config.get('tokenizer_class')!r}."
        )

    for key, (piece, expected_id) in EXPECTED_LLAMA_SPECIAL_TOKENS.items():
        config_id_key = {
            "unk_token": None,
            "bos_token": "bos_token_id",
            "eos_token": "eos_token_id",
            "pad_token": "pad_token_id",
        }[key]
        actual_piece = tokenizer_config.get(key)
        mapped_piece = special_tokens_map.get(key)
        actual_id = int(sp.piece_to_id(piece))
        if actual_piece != piece:
            raise ValueError(f"tokenizer_config.json {key} muss {piece!r} sein, ist {actual_piece!r}.")
        if mapped_piece != piece:
            raise ValueError(f"special_tokens_map.json {key} muss {piece!r} sein, ist {mapped_piece!r}.")
        if actual_id != expected_id:
            raise ValueError(f"tokenizer.model {piece!r} muss ID {expected_id} haben, hat ID {actual_id}.")
        if config_id_key and int(config.get(config_id_key, -1)) != expected_id:
            raise ValueError(
                f"config.json {config_id_key} muss {expected_id} sein, ist {config.get(config_id_key)!r}."
            )

    if not sp.is_unknown(0):
        raise ValueError("tokenizer.model ID 0 muss UNKNOWN sein.")
    if not sp.is_control(1) or not sp.is_control(2) or not sp.is_control(3):
        raise ValueError("BOS, EOS und PAD muessen im tokenizer.model Control-Tokens sein.")

    byte_pieces = [sp.id_to_piece(i) for i in range(piece_size) if sp.is_byte(i)]
    if byte_pieces:
        preview = ", ".join(byte_pieces[:5])
        raise ValueError(
            "Byte-Fallback-Tokens im SentencePiece-Modell gefunden. "
            f"Fuer diesen LLaMA/GGUF-Smoke-Test byte_fallback=false verwenden. Beispiele: {preview}"
        )


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        shutil.copy2(source, target)


def prepare_staging_model(model_dir: str | Path, staging_dir: str | Path) -> Path:
    root = Path(model_dir)
    staging = Path(staging_dir)
    config_file, weight_files, tokenizer_dir = validate_model_dir(root)

    staging.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_file, staging / "config.json")
    copy_if_exists(root / "generation_config.json", staging / "generation_config.json")

    for weight_file in weight_files:
        shutil.copy2(weight_file, staging / weight_file.name)

    tokenizer_files = [
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ]
    for filename in tokenizer_files:
        copy_if_exists(tokenizer_dir / filename, staging / filename)

    LOGGER.info("HF-Staging fuer llama.cpp vorbereitet: %s", staging)
    LOGGER.info("Tokenizer-Dateien stammen aus: %s", tokenizer_dir)
    return staging


def build_convert_command(
    python_executable: str,
    converter: Path,
    staged_model_dir: Path,
    output_file: Path,
    outtype: str,
) -> list[str]:
    return [
        python_executable,
        str(converter),
        str(staged_model_dir),
        "--outfile",
        str(output_file),
        "--outtype",
        outtype,
    ]


def export_gguf(
    model_dir: str | Path,
    output_file: str | Path,
    llama_cpp_dir: str | Path,
    outtype: str = "f16",
    python_executable: str | None = None,
    keep_staging: bool = False,
) -> Path:
    model_path = Path(model_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter = find_llama_converter(llama_cpp_dir)
    python_bin = python_executable or sys.executable

    LOGGER.info("Pruefe lokales HF-Modell: %s", model_path)
    validate_model_dir(model_path)
    LOGGER.info("llama.cpp-Konverter: %s", converter)

    if keep_staging:
        staging_path = output_path.with_suffix(".hf-staging")
        if staging_path.exists():
            shutil.rmtree(staging_path)
        prepare_staging_model(model_path, staging_path)
        command = build_convert_command(python_bin, converter, staging_path, output_path, outtype)
        LOGGER.info("Starte GGUF-Export: %s", " ".join(command))
        subprocess.run(command, check=True)
    else:
        with tempfile.TemporaryDirectory(prefix="lumen-gguf-") as temp_dir:
            staging_path = prepare_staging_model(model_path, Path(temp_dir) / "hf-model")
            command = build_convert_command(python_bin, converter, staging_path, output_path, outtype)
            LOGGER.info("Starte GGUF-Export: %s", " ".join(command))
            subprocess.run(command, check=True)

    if not output_path.exists():
        raise FileNotFoundError(f"GGUF-Datei wurde nicht erzeugt: {output_path}")
    LOGGER.info("GGUF exportiert: %s", output_path)
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exportiert models/smoke/final nach GGUF.")
    parser.add_argument("--model-dir", default="models/smoke/final", help="Lokales HF-Modellverzeichnis.")
    parser.add_argument(
        "--output-file",
        default="models/smoke/quantum-smoke-f16.gguf",
        help="Zieldatei fuer GGUF.",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        default="/srv/lumen/llama.cpp",
        help="Pfad zu llama.cpp mit convert_hf_to_gguf.py.",
    )
    parser.add_argument("--outtype", default="f16", choices=["f16", "f32", "bf16"], help="GGUF-Ausgabetyp.")
    parser.add_argument(
        "--python",
        dest="python_executable",
        help="Python-Executable fuer den llama.cpp-Konverter. Standard: aktueller Python-Interpreter.",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="HF-Staging-Ordner neben der GGUF-Datei behalten, nuetzlich zum Debuggen.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    export_gguf(
        model_dir=args.model_dir,
        output_file=args.output_file,
        llama_cpp_dir=args.llama_cpp_dir,
        outtype=args.outtype,
        python_executable=args.python_executable,
        keep_staging=args.keep_staging,
    )


if __name__ == "__main__":
    main()
