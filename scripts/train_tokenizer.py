"""Train the local SentencePiece BPE tokenizer for rappidAI Quantum.

Dieses Skript nutzt nur lokale Textdateien aus data/raw und speichert einen
LLaMA-/llama.cpp-kompatiblen Tokenizer unter tokenizer/smoke. Es werden keine
externen Tokenizer oder Modellgewichte geladen.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import yaml

try:
    import sentencepiece as spm
except ImportError as exc:  # pragma: no cover - exercised only without dependency installed.
    raise ImportError(
        "sentencepiece ist erforderlich fuer den LLaMA-/GGUF-kompatiblen Tokenizer. "
        "Installiere zuerst: pip install -r requirements.txt"
    ) from exc


LOGGER = logging.getLogger("lumen.train_tokenizer")

DEFAULT_SPECIAL_TOKENS = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
}


class SentencePieceLlamaTokenizer:
    """Minimaler lokaler Tokenizer-Adapter fuer Training und Generierung."""

    def __init__(
        self,
        model_file: str | Path,
        special_tokens: dict[str, str] | None = None,
        model_max_length: int | None = None,
    ):
        self.model_file = str(model_file)
        self.special_tokens = {**DEFAULT_SPECIAL_TOKENS, **(special_tokens or {})}
        self.sp_model = spm.SentencePieceProcessor(model_file=self.model_file)
        self.model_max_length = model_max_length
        self.unk_token = self.special_tokens["unk_token"]
        self.bos_token = self.special_tokens["bos_token"]
        self.eos_token = self.special_tokens["eos_token"]
        self.pad_token = self.special_tokens["pad_token"]
        self.unk_token_id = self.sp_model.piece_to_id(self.unk_token)
        self.bos_token_id = self.sp_model.piece_to_id(self.bos_token)
        self.eos_token_id = self.sp_model.piece_to_id(self.eos_token)
        self.pad_token_id = self.sp_model.piece_to_id(self.pad_token)

    def __len__(self) -> int:
        return int(self.sp_model.get_piece_size())

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = list(self.sp_model.encode(text, out_type=int))
        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        ids = [int(token_id) for token_id in token_ids]
        if skip_special_tokens:
            special_ids = {
                self.unk_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.pad_token_id,
            }
            ids = [token_id for token_id in ids if token_id not in special_ids]
        return self.sp_model.decode(ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(self.sp_model.piece_to_id(token))

    def __call__(
        self, text: str, return_tensors: str | None = None, add_special_tokens: bool = False
    ):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if return_tensors == "pt":
            import torch

            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        if return_tensors is not None:
            raise ValueError(f"return_tensors={return_tensors!r} wird nicht unterstuetzt.")
        return {"input_ids": ids}


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


def load_fast_tokenizer(
    tokenizer_dir: str | Path,
    special_tokens: dict[str, str] | None = None,
    model_max_length: int | None = None,
) -> SentencePieceLlamaTokenizer:
    """Laedt den lokalen SentencePiece-Tokenizer ohne from_pretrained."""

    tokens = {**DEFAULT_SPECIAL_TOKENS, **(special_tokens or {})}
    tokenizer_file = Path(tokenizer_dir) / "tokenizer.model"
    if not tokenizer_file.exists():
        raise FileNotFoundError(
            f"SentencePiece-Tokenizer nicht gefunden: {tokenizer_file}. "
            "Fuehre zuerst train_tokenizer.py aus."
        )

    tokenizer = SentencePieceLlamaTokenizer(
        model_file=tokenizer_file,
        special_tokens=tokens,
        model_max_length=model_max_length,
    )
    # Nur setzen, wenn ausdruecklich gewuenscht. Beim Tokenisieren der Rohdaten
    # bleibt der Wert bewusst offen, damit lange Absaetze nicht abgeschnitten,
    # sondern in make_blocks sauber in Bloecke gepackt werden.
    if model_max_length is not None:
        tokenizer.model_max_length = int(model_max_length)
    return tokenizer


def write_training_corpus(files: list[Path]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        prefix="lumen_sentencepiece_",
        delete=False,
    )
    with handle:
        for path in files:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                handle.write(text)
                handle.write("\n")
    return Path(handle.name)


def train_tokenizer(
    input_dir: str | Path,
    output_dir: str | Path,
    vocab_size: int,
    min_frequency: int,
    seed: int,
    special_tokens: dict[str, str] | None = None,
    model_max_length: int | None = None,
    byte_fallback: bool = False,
) -> SentencePieceLlamaTokenizer:
    # Hinweis zur Reproduzierbarkeit: SentencePiece trainiert deterministisch,
    # solange die Eingabetexte in stabiler Reihenfolge zusammengefuehrt werden.
    # Der Seed wird zur Nachvollziehbarkeit dokumentiert.
    tokens = {**DEFAULT_SPECIAL_TOKENS, **(special_tokens or {})}
    files = find_text_files(input_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Trainiere SentencePiece-BPE-Tokenizer aus %d Textdatei(en).", len(files))
    corpus_file = write_training_corpus(files)
    model_prefix = output / "tokenizer"
    try:
        spm.SentencePieceTrainer.Train(
            input=str(corpus_file),
            model_prefix=str(model_prefix),
            model_type="bpe",
            vocab_size=int(vocab_size),
            character_coverage=1.0,
            input_sentence_size=0,
            shuffle_input_sentence=False,
            hard_vocab_limit=False,
            byte_fallback=bool(byte_fallback),
            split_digits=True,
            allow_whitespace_only_pieces=True,
            remove_extra_whitespaces=False,
            normalization_rule_name="nfkc",
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            unk_piece=tokens["unk_token"],
            bos_piece=tokens["bos_token"],
            eos_piece=tokens["eos_token"],
            pad_piece=tokens["pad_token"],
            minloglevel=1,
        )
    finally:
        corpus_file.unlink(missing_ok=True)

    fast_tokenizer = load_fast_tokenizer(output, tokens, model_max_length=model_max_length)

    tokenizer_config = output / "tokenizer_config.json"
    config = {
        "tokenizer_class": "LlamaTokenizer",
        "model_max_length": model_max_length,
        "bos_token": tokens["bos_token"],
        "eos_token": tokens["eos_token"],
        "pad_token": tokens["pad_token"],
        "unk_token": tokens["unk_token"],
        "add_bos_token": True,
        "add_eos_token": False,
        "legacy": False,
    }
    tokenizer_config.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    special_tokens_map = {
        "bos_token": tokens["bos_token"],
        "eos_token": tokens["eos_token"],
        "pad_token": tokens["pad_token"],
        "unk_token": tokens["unk_token"],
    }
    (output / "special_tokens_map.json").write_text(
        json.dumps(special_tokens_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "input_dir": str(Path(input_dir)),
        "input_files": [str(path) for path in files],
        "output_dir": str(output),
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": len(fast_tokenizer),
        "min_frequency": min_frequency,
        "special_tokens": tokens,
        "tokenizer_library": "sentencepiece",
        "model_type": "bpe",
        "normalizer": "NFKC",
        "byte_fallback": bool(byte_fallback),
        "sentencepiece_model": str(output / "tokenizer.model"),
        "model_max_length": model_max_length,
    }
    with (output / "tokenizer_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    LOGGER.info(
        "SentencePiece-Tokenizer gespeichert in %s mit %d Tokens.",
        output,
        len(fast_tokenizer),
    )
    return fast_tokenizer


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trainiert den Lumen-Smoke-SentencePiece-BPE-Tokenizer."
    )
    parser.add_argument(
        "--config", default="configs/smoke_5m.yaml", help="Pfad zur YAML-Konfiguration."
    )
    parser.add_argument("--input-dir", help="Ordner mit .txt-Rohdaten. Ueberschreibt die Config.")
    parser.add_argument(
        "--output-dir", help="Zielordner fuer den Tokenizer. Ueberschreibt die Config."
    )
    parser.add_argument(
        "--vocab-size", type=int, help="BPE-Vokabulargroesse. Ueberschreibt die Config."
    )
    parser.add_argument("--min-frequency", type=int, help="Mindesthaeufigkeit fuer BPE-Merges.")
    parser.add_argument(
        "--seed", type=int, help="Seed fuer reproduzierbare Dateireihenfolge/Einstellungen."
    )
    parser.add_argument(
        "--byte-fallback",
        action="store_true",
        help="Byte-Fallback aktivieren. Fuer llama.cpp-Smoke standardmaessig aus.",
    )
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
        byte_fallback=bool(args.byte_fallback or tokenizer_config.get("byte_fallback", False)),
    )


if __name__ == "__main__":
    main()
