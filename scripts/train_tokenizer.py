"""Trainiert den lokalen BPE-Tokenizer fuer Lumen Quantum.

Dieses Skript nutzt nur lokale Textdateien aus data/raw und speichert einen
Hugging-Face-kompatiblen FastTokenizer unter tokenizer/smoke. Es werden keine
externen Tokenizer oder Modellgewichte geladen.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


LOGGER = logging.getLogger("lumen.train_tokenizer")

DEFAULT_SPECIAL_TOKENS = {
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
    "pad_token": "<|pad|>",
    "unk_token": "<|unk|>",
}


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


def find_text_files(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Rohdatenordner nicht gefunden: {root}. Lege zuerst .txt-Dateien in data/raw/ ab."
        )

    files = sorted(path for path in root.rglob("*.txt") if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"Keine .txt-Dateien in {root} gefunden. Erstelle zuerst Trainingsdaten in data/raw/."
        )
    return files


def make_tokenizer(unk_token: str) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def load_fast_tokenizer(
    tokenizer_dir: str | Path,
    special_tokens: dict[str, str] | None = None,
    model_max_length: int | None = None,
) -> PreTrainedTokenizerFast:
    """Laedt den lokalen Tokenizer ohne AutoTokenizer.from_pretrained."""

    tokens = {**DEFAULT_SPECIAL_TOKENS, **(special_tokens or {})}
    tokenizer_file = Path(tokenizer_dir) / "tokenizer.json"
    if not tokenizer_file.exists():
        raise FileNotFoundError(
            f"Tokenizer-Datei nicht gefunden: {tokenizer_file}. Fuehre zuerst train_tokenizer.py aus."
        )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        bos_token=tokens["bos_token"],
        eos_token=tokens["eos_token"],
        pad_token=tokens["pad_token"],
        unk_token=tokens["unk_token"],
    )
    # Nur setzen, wenn ausdruecklich gewuenscht. Beim Tokenisieren der Rohdaten
    # bleibt der Wert bewusst offen, damit lange Absaetze nicht abgeschnitten,
    # sondern in make_blocks sauber in Bloecke gepackt werden.
    if model_max_length is not None:
        tokenizer.model_max_length = int(model_max_length)
    return tokenizer


def train_tokenizer(
    input_dir: str | Path,
    output_dir: str | Path,
    vocab_size: int,
    min_frequency: int,
    seed: int,
    special_tokens: dict[str, str] | None = None,
    model_max_length: int | None = None,
) -> PreTrainedTokenizerFast:
    # Hinweis zur Reproduzierbarkeit: Das BPE-Training der tokenizers-Bibliothek
    # ist deterministisch, sobald die Eingabedateien in stabiler Reihenfolge
    # vorliegen (siehe sortiertes find_text_files). Der Seed selbst beeinflusst
    # das Rust-Training nicht; er wird nur zur Nachvollziehbarkeit dokumentiert.
    tokens = {**DEFAULT_SPECIAL_TOKENS, **(special_tokens or {})}
    special_token_values = [
        tokens["bos_token"],
        tokens["eos_token"],
        tokens["pad_token"],
        tokens["unk_token"],
    ]

    files = find_text_files(input_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Trainiere BPE-Tokenizer aus %d Textdatei(en).", len(files))
    tokenizer = make_tokenizer(tokens["unk_token"])
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_token_values,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train(files=[str(path) for path in files], trainer=trainer)
    tokenizer.save(str(output / "tokenizer.json"))

    fast_tokenizer = load_fast_tokenizer(output, tokens, model_max_length=model_max_length)
    fast_tokenizer.save_pretrained(str(output))

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "input_dir": str(Path(input_dir)),
        "input_files": [str(path) for path in files],
        "output_dir": str(output),
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": len(fast_tokenizer),
        "min_frequency": min_frequency,
        "special_tokens": tokens,
        "normalizer": "NFKC",
        "pre_tokenizer": "ByteLevel(add_prefix_space=True)",
        "model_max_length": model_max_length,
    }
    with (output / "tokenizer_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    LOGGER.info("Tokenizer gespeichert in %s mit %d Tokens.", output, len(fast_tokenizer))
    return fast_tokenizer


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trainiert den Lumen-Smoke-BPE-Tokenizer.")
    parser.add_argument("--config", default="configs/smoke_5m.yaml", help="Pfad zur YAML-Konfiguration.")
    parser.add_argument("--input-dir", help="Ordner mit .txt-Rohdaten. Ueberschreibt die Config.")
    parser.add_argument("--output-dir", help="Zielordner fuer den Tokenizer. Ueberschreibt die Config.")
    parser.add_argument("--vocab-size", type=int, help="BPE-Vokabulargroesse. Ueberschreibt die Config.")
    parser.add_argument("--min-frequency", type=int, help="Mindesthaeufigkeit fuer BPE-Merges.")
    parser.add_argument("--seed", type=int, help="Seed fuer reproduzierbare Dateireihenfolge/Einstellungen.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    config = load_config(args.config)
    tokenizer_config = config["tokenizer"]
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    # Kontextlaenge des Tokenizers = Modell-Kontextlaenge, damit sie nicht
    # hartkodiert ist und bei geaenderter block_size konsistent bleibt.
    context_length = int(
        model_config.get("max_position_embeddings", data_config.get("block_size", 256))
    )

    train_tokenizer(
        input_dir=args.input_dir or tokenizer_config["input_dir"],
        output_dir=args.output_dir or tokenizer_config["output_dir"],
        vocab_size=args.vocab_size or int(tokenizer_config["vocab_size"]),
        min_frequency=args.min_frequency or int(tokenizer_config["min_frequency"]),
        seed=args.seed if args.seed is not None else int(config["seed"]),
        special_tokens=tokenizer_config.get("special_tokens"),
        model_max_length=context_length,
    )


if __name__ == "__main__":
    main()
