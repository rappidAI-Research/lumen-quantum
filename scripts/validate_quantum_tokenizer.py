"""Validiert den quantum-1 Pilot-Tokenizer vor spaeterem Modelltraining/GGUF."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import yaml

try:
    import sentencepiece as spm
except ImportError as exc:  # pragma: no cover - depends on local environment.
    raise ImportError(
        "sentencepiece ist fuer die Tokenizer-Validierung erforderlich. "
        "Installiere zuerst: pip install -r requirements.txt"
    ) from exc


LOGGER = logging.getLogger("lumen.validate_quantum_tokenizer")

REQUIRED_FILES = [
    "tokenizer.model",
    "tokenizer.vocab",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer_manifest.json",
]

EXPECTED_BASE_IDS = {
    "unk_token": ("<unk>", 0),
    "bos_token": ("<s>", 1),
    "eos_token": ("</s>", 2),
    "pad_token": ("<pad>", 3),
}

DEFAULT_GERMAN_SENTENCES = [
    "Äpfel, Öl und Grüße aus Köln.",
    "Fußgänger überqueren die Straße.",
    "Lumen verarbeitet deutsche Umlaute zuverlässig.",
]


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


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_files(tokenizer_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (tokenizer_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Tokenizer-Dateien fehlen in {tokenizer_dir}: {', '.join(missing)}. "
            "Fuehre zuerst scripts/train_quantum_tokenizer.py aus."
        )


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise ValueError(f"{message}: erwartet {expected!r}, erhalten {actual!r}")


def validate_special_tokens(
    sp: spm.SentencePieceProcessor,
    tokenizer_config: dict,
    special_tokens_map: dict,
    manifest: dict,
    chat_tokens: list[str],
) -> dict[str, int]:
    ids: dict[str, int] = {}
    for key, (piece, expected_id) in EXPECTED_BASE_IDS.items():
        assert_equal(tokenizer_config.get(key), piece, f"tokenizer_config.json {key}")
        assert_equal(special_tokens_map.get(key), piece, f"special_tokens_map.json {key}")
        actual_id = int(sp.piece_to_id(piece))
        assert_equal(actual_id, expected_id, f"tokenizer.model ID fuer {piece}")
        ids[piece] = actual_id

    if not sp.is_unknown(0):
        raise ValueError("tokenizer.model ID 0 muss UNKNOWN sein.")
    if not sp.is_control(1) or not sp.is_control(2) or not sp.is_control(3):
        raise ValueError("BOS, EOS und PAD muessen SentencePiece-Control-Tokens sein.")

    config_chat_tokens = list(tokenizer_config.get("additional_special_tokens") or [])
    map_chat_tokens = list(special_tokens_map.get("additional_special_tokens") or [])
    assert_equal(config_chat_tokens, chat_tokens, "tokenizer_config.json additional_special_tokens")
    assert_equal(map_chat_tokens, chat_tokens, "special_tokens_map.json additional_special_tokens")

    for token in chat_tokens:
        token_id = int(sp.piece_to_id(token))
        if token_id < 0 or token_id == int(sp.unk_id()):
            raise ValueError(f"Chat-Sondertoken fehlt im SentencePiece-Modell: {token}")
        ids[token] = token_id

    manifest_ids = manifest.get("token_ids", {})
    for token, token_id in ids.items():
        assert_equal(
            int(manifest_ids.get(token, -1)),
            token_id,
            f"tokenizer_manifest.json token_ids[{token}]",
        )

    return ids


def validate_no_byte_fallback(sp: spm.SentencePieceProcessor) -> None:
    byte_tokens = [
        sp.id_to_piece(index) for index in range(sp.get_piece_size()) if sp.is_byte(index)
    ]
    if byte_tokens:
        preview = ", ".join(byte_tokens[:5])
        raise ValueError(
            "Byte-Fallback-Tokens gefunden. Fuer diesen LLaMA/GGUF-Pilot muss byte_fallback=false bleiben. "
            f"Beispiele: {preview}"
        )


def validate_roundtrip_and_umlauts(sp: spm.SentencePieceProcessor, sentences: list[str]) -> None:
    required_chars = {"ä", "ö", "ü", "ß"}
    joined = " ".join(sentences)
    missing = sorted(char for char in required_chars if char not in joined.lower())
    if missing:
        raise ValueError(
            f"Validierungssaetze muessen deutsche Umlaute enthalten, fehlend: {missing}"
        )

    unk_id = int(sp.unk_id())
    for sentence in sentences:
        ids = list(sp.encode(sentence, out_type=int))
        if not ids:
            raise ValueError(f"Leeres Encoding fuer Satz: {sentence!r}")
        if unk_id in ids:
            raise ValueError(f"UNK-Token in normalem deutschem Satz gefunden: {sentence!r}")
        decoded = sp.decode(ids).strip()
        if decoded != sentence.strip():
            raise ValueError(f"Encode/decode-Roundtrip fehlgeschlagen: {sentence!r} -> {decoded!r}")


def validate_quantum_tokenizer(config_path: str | Path) -> dict:
    config = load_config(config_path)
    tokenizer_dir = Path(config["tokenizer"]["output_dir"])
    require_files(tokenizer_dir)

    tokenizer_config = read_json(tokenizer_dir / "tokenizer_config.json")
    special_tokens_map = read_json(tokenizer_dir / "special_tokens_map.json")
    manifest = read_json(tokenizer_dir / "tokenizer_manifest.json")
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_dir / "tokenizer.model"))

    assert_equal(tokenizer_config.get("tokenizer_class"), "LlamaTokenizer", "tokenizer_class")
    actual_vocab_size = int(sp.get_piece_size())
    expected_vocab_size = int(config["tokenizer"]["vocab_size"])
    future_config = config["future_llama_config"]
    future_vocab_size = int(future_config["vocab_size"])
    assert_equal(actual_vocab_size, expected_vocab_size, "tokenizer.model vocab_size")
    assert_equal(actual_vocab_size, future_vocab_size, "future_llama_config vocab_size")
    assert_equal(int(future_config["unk_token_id"]), 0, "future_llama_config unk_token_id")
    assert_equal(int(future_config["bos_token_id"]), 1, "future_llama_config bos_token_id")
    assert_equal(int(future_config["eos_token_id"]), 2, "future_llama_config eos_token_id")
    assert_equal(int(future_config["pad_token_id"]), 3, "future_llama_config pad_token_id")
    assert_equal(
        int(manifest.get("actual_vocab_size", -1)), actual_vocab_size, "Manifest actual_vocab_size"
    )
    assert_equal(
        int(manifest.get("future_llama_vocab_size", -1)),
        future_vocab_size,
        "Manifest future_llama_vocab_size",
    )

    chat_tokens = list(config["tokenizer"]["special_tokens"].get("additional_special_tokens") or [])
    token_id_map = validate_special_tokens(
        sp, tokenizer_config, special_tokens_map, manifest, chat_tokens
    )
    validate_no_byte_fallback(sp)

    validation_config = config.get("validation", {})
    sentences = list(validation_config.get("normal_sentences") or DEFAULT_GERMAN_SENTENCES)
    sentences.extend(validation_config.get("roundtrip_sentences") or [])
    validate_roundtrip_and_umlauts(sp, sentences)

    report = {
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "tokenizer_dir": str(tokenizer_dir),
        "actual_vocab_size": actual_vocab_size,
        "future_llama_vocab_size": future_vocab_size,
        "tokenizer_class": tokenizer_config.get("tokenizer_class"),
        "token_ids": token_id_map,
        "sentencepiece_version": getattr(spm, "__version__", "unknown"),
        "gguf_pre_export_checks": {
            "has_tokenizer_model": True,
            "has_tokenizer_config": True,
            "has_special_tokens_map": True,
            "no_byte_fallback_tokens": True,
            "classic_llama_ids": True,
        },
    }
    (tokenizer_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Tokenizer-Validierung erfolgreich: %s", tokenizer_dir)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validiert den quantum-1 Pilot-Tokenizer.")
    parser.add_argument(
        "--config",
        default="configs/quantum_1_tokenizer_pilot.yaml",
        help="Pfad zur YAML-Konfiguration.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    validate_quantum_tokenizer(args.config)


if __name__ == "__main__":
    main()
