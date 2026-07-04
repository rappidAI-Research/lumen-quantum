"""Trainiert einen quantum-1 SentencePiece-BPE-Tokenizer aus lokalen Trainingsdaten.

Der Tokenizer wird vollstaendig selbst mit SentencePiece-BPE trainiert. Das
Skript nutzt ausschliesslich train.jsonl und laedt keine vortrainierten
Tokenizer oder Modellgewichte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

try:
    import sentencepiece as spm
except ImportError as exc:  # pragma: no cover - depends on local environment.
    raise ImportError(
        "sentencepiece ist fuer den quantum-1 Pilot-Tokenizer erforderlich. "
        "Installiere zuerst: pip install -r requirements.txt"
    ) from exc


LOGGER = logging.getLogger("lumen.train_quantum_tokenizer")

BASE_SPECIAL_TOKEN_IDS = {
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    "<pad>": 3,
}

DEFAULT_CHAT_SPECIAL_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>"]

GENERATED_FILES = [
    "tokenizer.model",
    "tokenizer.vocab",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer_manifest.json",
    "validation_report.json",
]

FREEZE_MARKER_FILE = "FINAL_FROZEN"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_train_only_path(train_file: str | Path) -> Path:
    path = Path(train_file)
    lowered_name = path.name.lower()
    if lowered_name != "train.jsonl":
        raise ValueError(
            "Der quantum-1 Tokenizer darf nur auf train.jsonl trainiert werden. "
            f"Erhalten: {path}"
        )
    if any(forbidden in str(path).lower() for forbidden in ("validation.jsonl", "test.jsonl")):
        raise ValueError(f"Validation/Test duerfen nicht fuer Tokenizer-Training genutzt werden: {path}")
    if not path.exists():
        raise FileNotFoundError(
            f"Trainingsdatei nicht gefunden: {path}. "
            "Fuehre zuerst die quantum-1 Datenpipeline bis sample_quantum_data.py aus."
        )
    return path


def iter_jsonl_texts(path: str | Path, text_field: str) -> Iterable[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ungueltiges JSONL in {path}:{line_number}") from exc
            text = record.get(text_field)
            if not isinstance(text, str):
                continue
            text = text.strip()
            if text:
                yield text


def write_sentencepiece_corpus(train_file: Path, text_field: str) -> tuple[Path, int, int]:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        prefix="lumen_quantum_tokenizer_",
        delete=False,
    )
    document_count = 0
    text_bytes = 0
    with handle:
        for text in iter_jsonl_texts(train_file, text_field):
            document_count += 1
            text_bytes += len(text.encode("utf-8"))
            handle.write(text.replace("\r\n", "\n").replace("\r", "\n"))
            handle.write("\n")

    if document_count == 0:
        Path(handle.name).unlink(missing_ok=True)
        raise ValueError(f"Keine verwendbaren Texte im Feld {text_field!r} gefunden: {train_file}")

    return Path(handle.name), document_count, text_bytes


def remove_previous_outputs(output_dir: Path, freeze_marker_file: str = FREEZE_MARKER_FILE) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_marker = output_dir / freeze_marker_file
    if freeze_marker.exists():
        raise FileExistsError(
            f"Tokenizer ist eingefroren und wird nicht ueberschrieben: {freeze_marker}. "
            "Lege fuer eine neue finale Version einen neuen Tokenizer-Ordner an."
        )
    for filename in GENERATED_FILES:
        (output_dir / filename).unlink(missing_ok=True)


def get_special_tokens(config: dict) -> tuple[dict[str, str], list[str]]:
    special = config["tokenizer"]["special_tokens"]
    base_tokens = {
        "unk_token": special.get("unk_token", "<unk>"),
        "bos_token": special.get("bos_token", "<s>"),
        "eos_token": special.get("eos_token", "</s>"),
        "pad_token": special.get("pad_token", "<pad>"),
    }
    chat_tokens = list(special.get("additional_special_tokens") or DEFAULT_CHAT_SPECIAL_TOKENS)
    return base_tokens, chat_tokens


def train_sentencepiece(
    corpus_file: Path,
    output_dir: Path,
    config: dict,
    base_tokens: dict[str, str],
    chat_tokens: list[str],
) -> None:
    tokenizer_config = config["tokenizer"]
    model_prefix = output_dir / "tokenizer"

    spm.SentencePieceTrainer.Train(
        input=str(corpus_file),
        model_prefix=str(model_prefix),
        model_type="bpe",
        vocab_size=int(tokenizer_config["vocab_size"]),
        character_coverage=float(tokenizer_config.get("character_coverage", 1.0)),
        input_sentence_size=0,
        shuffle_input_sentence=False,
        hard_vocab_limit=bool(tokenizer_config.get("hard_vocab_limit", True)),
        byte_fallback=bool(tokenizer_config.get("byte_fallback", False)),
        split_digits=bool(tokenizer_config.get("split_digits", True)),
        allow_whitespace_only_pieces=bool(tokenizer_config.get("allow_whitespace_only_pieces", True)),
        remove_extra_whitespaces=bool(tokenizer_config.get("remove_extra_whitespaces", False)),
        normalization_rule_name=str(tokenizer_config.get("normalization_rule_name", "nfkc")),
        user_defined_symbols=",".join(chat_tokens),
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_piece=base_tokens["unk_token"],
        bos_piece=base_tokens["bos_token"],
        eos_piece=base_tokens["eos_token"],
        pad_piece=base_tokens["pad_token"],
        minloglevel=1,
    )


def load_sentencepiece(path: Path) -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=str(path))


def token_ids(sp: spm.SentencePieceProcessor, tokens: Iterable[str]) -> dict[str, int]:
    return {token: int(sp.piece_to_id(token)) for token in tokens}


def write_hf_tokenizer_files(
    output_dir: Path,
    config: dict,
    base_tokens: dict[str, str],
    chat_tokens: list[str],
) -> None:
    tokenizer_config = {
        "tokenizer_class": "LlamaTokenizer",
        "model_max_length": config.get("future_llama_config", {}).get("max_position_embeddings"),
        "bos_token": base_tokens["bos_token"],
        "eos_token": base_tokens["eos_token"],
        "pad_token": base_tokens["pad_token"],
        "unk_token": base_tokens["unk_token"],
        "additional_special_tokens": chat_tokens,
        "add_bos_token": True,
        "add_eos_token": False,
        "legacy": False,
        "clean_up_tokenization_spaces": False,
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    special_tokens_map = {
        "bos_token": base_tokens["bos_token"],
        "eos_token": base_tokens["eos_token"],
        "pad_token": base_tokens["pad_token"],
        "unk_token": base_tokens["unk_token"],
        "additional_special_tokens": chat_tokens,
    }
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(special_tokens_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_manifest(
    output_dir: Path,
    config: dict,
    config_path: Path,
    train_file: Path,
    document_count: int,
    text_bytes: int,
    base_tokens: dict[str, str],
    chat_tokens: list[str],
) -> dict:
    sp = load_sentencepiece(output_dir / "tokenizer.model")
    all_special_tokens = [
        base_tokens["unk_token"],
        base_tokens["bos_token"],
        base_tokens["eos_token"],
        base_tokens["pad_token"],
        *chat_tokens,
    ]
    manifest = {
        "tokenizer_name": config.get("project", {}).get("tokenizer_name", "quantum-1-pilot"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path),
        "training_file": str(train_file),
        "training_data_sha256": sha256_file(train_file),
        "training_document_count": document_count,
        "training_text_bytes": text_bytes,
        "requested_vocab_size": int(config["tokenizer"]["vocab_size"]),
        "actual_vocab_size": int(sp.get_piece_size()),
        "future_llama_vocab_size": int(config["future_llama_config"]["vocab_size"]),
        "token_ids": token_ids(sp, all_special_tokens),
        "expected_special_token_order": all_special_tokens,
        "base_special_tokens": base_tokens,
        "additional_special_tokens": chat_tokens,
        "sentencepiece_version": getattr(spm, "__version__", "unknown"),
        "seed": int(config["seed"]),
        "sentencepiece": {
            "model_type": "bpe",
            "normalization_rule_name": config["tokenizer"].get("normalization_rule_name", "nfkc"),
            "byte_fallback": bool(config["tokenizer"].get("byte_fallback", False)),
            "hard_vocab_limit": bool(config["tokenizer"].get("hard_vocab_limit", True)),
        },
        "gguf_compatibility": {
            "tokenizer_class": "LlamaTokenizer",
            "tokenizer_model_file": "tokenizer.model",
            "uses_pretrained_tokenizer": False,
            "classic_llama_special_token_ids": BASE_SPECIAL_TOKEN_IDS,
        },
    }
    (output_dir / "tokenizer_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def train_quantum_tokenizer(config_path: str | Path) -> Path:
    config_file = Path(config_path)
    config = load_config(config_file)
    train_file = ensure_train_only_path(config["data"]["train_file"])
    output_dir = Path(config["tokenizer"]["output_dir"])
    base_tokens, chat_tokens = get_special_tokens(config)

    remove_previous_outputs(
        output_dir,
        str(config["tokenizer"].get("freeze_marker_file", FREEZE_MARKER_FILE)),
    )
    corpus_file, document_count, text_bytes = write_sentencepiece_corpus(
        train_file=train_file,
        text_field=str(config["data"].get("text_field", "text")),
    )
    try:
        tokenizer_name = config.get("project", {}).get("tokenizer_name", "quantum-1")
        LOGGER.info("Trainiere %s Tokenizer aus %s", tokenizer_name, train_file)
        LOGGER.info("Trainingsdokumente: %d, Textbytes: %d", document_count, text_bytes)
        train_sentencepiece(corpus_file, output_dir, config, base_tokens, chat_tokens)
    finally:
        corpus_file.unlink(missing_ok=True)

    write_hf_tokenizer_files(output_dir, config, base_tokens, chat_tokens)
    manifest = write_manifest(
        output_dir=output_dir,
        config=config,
        config_path=config_file,
        train_file=train_file,
        document_count=document_count,
        text_bytes=text_bytes,
        base_tokens=base_tokens,
        chat_tokens=chat_tokens,
    )
    LOGGER.info(
        "Tokenizer gespeichert in %s mit %d Tokens.",
        output_dir,
        manifest["actual_vocab_size"],
    )
    return output_dir


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trainiert einen quantum-1 SentencePiece-BPE-Tokenizer.")
    parser.add_argument(
        "--config",
        default="configs/quantum_1_tokenizer_pilot.yaml",
        help="Pfad zur YAML-Konfiguration.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    train_quantum_tokenizer(args.config)


if __name__ == "__main__":
    main()
