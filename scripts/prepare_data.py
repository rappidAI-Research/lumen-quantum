"""Bereitet lokale Textdaten fuer das Smoke-Training vor."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml

try:
    from .train_tokenizer import load_fast_tokenizer
except ImportError:
    from train_tokenizer import load_fast_tokenizer


LOGGER = logging.getLogger("lumen.prepare_data")


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


def find_text_files(raw_dir: str | Path) -> list[Path]:
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Rohdatenordner nicht gefunden: {root}. Lege zuerst .txt-Dateien in data/raw/ ab."
        )
    files = sorted(path for path in root.rglob("*.txt") if path.is_file())
    if not files:
        raise FileNotFoundError(f"Keine .txt-Dateien in {root} gefunden.")
    return files


def read_text_units(raw_dir: str | Path) -> tuple[list[str], list[Path]]:
    files = find_text_files(raw_dir)
    units: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            LOGGER.warning("Ueberspringe leere Datei: %s", path)
            continue
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        units.extend(paragraphs or [text])

    if not units:
        raise ValueError(f"Alle .txt-Dateien in {raw_dir} sind leer.")
    return units, files


def deduplicate_units(units: list[str]) -> list[str]:
    """Entfernt exakte Duplikate, bevor gesplittet wird.

    Verhindert, dass identische Absaetze gleichzeitig in Train- und
    Validation-/Test-Split landen (Datenleck).
    """

    seen: set[str] = set()
    unique: list[str] = []
    for unit in units:
        if unit in seen:
            continue
        seen.add(unit)
        unique.append(unit)
    removed = len(units) - len(unique)
    if removed:
        LOGGER.info("Entferne %d exakte Duplikat-Texteinheiten vor dem Split.", removed)
    return unique


def split_units(
    units: list[str],
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str], str]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio muss zwischen 0 und 1 liegen.")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio muss zwischen 0 (inklusive) und 1 liegen.")
    if validation_ratio + test_ratio >= 1.0:
        raise ValueError("validation_ratio + test_ratio muss kleiner als 1 sein.")

    rng = random.Random(seed)
    shuffled = list(units)
    rng.shuffle(shuffled)
    n = len(shuffled)

    if n == 1:
        text = shuffled[0]
        length = len(text)
        test_len = int(length * test_ratio)
        val_len = max(1, int(length * validation_ratio))
        train_len = max(1, length - val_len - test_len)
        train_text = text[:train_len]
        val_text = text[train_len : train_len + val_len]
        test_text = text[train_len + val_len :]
        test_list = [test_text] if test_text.strip() else []
        return [train_text], [val_text], test_list, "single_unit_character_split"

    validation_count = max(1, int(round(n * validation_ratio)))
    test_count = max(1, int(round(n * test_ratio))) if test_ratio > 0 else 0
    # Immer mindestens ein Trainings-Unit behalten; bei sehr kleinen Datensaetzen
    # zuerst den Test-, dann den Validation-Split verkleinern.
    while n - validation_count - test_count < 1:
        if test_count > 1:
            test_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            test_count = 0
            break

    test_units = shuffled[:test_count]
    validation_units = shuffled[test_count : test_count + validation_count]
    train_units = shuffled[test_count + validation_count :]
    return train_units, validation_units, test_units, "shuffled_unit_split"


def encode_units(units: list[str], tokenizer) -> list[int]:
    ids: list[int] = []
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer muss BOS- und EOS-Sondertokens enthalten.")

    for text in units:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if token_ids:
            ids.extend([bos_id, *token_ids, eos_id])
    return ids


def make_blocks(
    token_ids: list[int], block_size: int, pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_size < 8:
        raise ValueError("block_size sollte mindestens 8 sein.")
    if not token_ids:
        raise ValueError("Keine Tokens erzeugt. Pruefe Rohdaten und Tokenizer.")

    blocks: list[list[int]] = []
    masks: list[list[int]] = []
    for start in range(0, len(token_ids), block_size):
        block = token_ids[start : start + block_size]
        real_len = len(block)
        if real_len < 2:
            # Ein Block mit weniger als 2 echten Tokens liefert nach dem
            # Causal-LM-Shift keine Trainingsziele (alle Labels -100) und damit
            # einen NaN-Loss. Solche Rest-Bloecke werden verworfen.
            continue
        mask = [1] * real_len
        if real_len < block_size:
            padding = block_size - real_len
            block = block + [pad_token_id] * padding
            mask = mask + [0] * padding
        blocks.append(block)
        masks.append(mask)

    if not blocks:
        raise ValueError(
            "Zu wenige Tokens fuer mindestens einen gueltigen Block (min. 2 echte Tokens noetig)."
        )

    return torch.tensor(blocks, dtype=torch.long), torch.tensor(masks, dtype=torch.long)


def prepare_data(
    raw_dir: str | Path,
    tokenizer_dir: str | Path,
    tokenized_dir: str | Path,
    processed_dir: str | Path,
    validation_ratio: float,
    test_ratio: float,
    block_size: int,
    seed: int,
    special_tokens: dict[str, str] | None = None,
) -> dict:
    tokenizer = load_fast_tokenizer(tokenizer_dir, special_tokens)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer muss ein PAD-Token besitzen.")

    units, files = read_text_units(raw_dir)
    units = deduplicate_units(units)
    train_units, validation_units, test_units, split_strategy = split_units(
        units, validation_ratio, test_ratio, seed
    )

    train_ids = encode_units(train_units, tokenizer)
    validation_ids = encode_units(validation_units, tokenizer)
    test_ids = encode_units(test_units, tokenizer)
    train_input_ids, train_attention_mask = make_blocks(
        train_ids, block_size, tokenizer.pad_token_id
    )
    val_input_ids, val_attention_mask = make_blocks(
        validation_ids, block_size, tokenizer.pad_token_id
    )

    tokenized_path = Path(tokenized_dir)
    processed_path = Path(processed_dir)
    tokenized_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    torch.save(
        {"input_ids": train_input_ids, "attention_mask": train_attention_mask},
        tokenized_path / "train.pt",
    )
    torch.save(
        {"input_ids": val_input_ids, "attention_mask": val_attention_mask},
        tokenized_path / "validation.pt",
    )

    # Der Test-Split wird als eigenes, waehrend des Trainings nie geladenes
    # Held-out-Set gespeichert. Bei extrem kleinen Datensaetzen kann er leer sein.
    test_blocks = 0
    test_path = tokenized_path / "test.pt"
    if len(test_ids) >= 2:
        test_input_ids, test_attention_mask = make_blocks(
            test_ids, block_size, tokenizer.pad_token_id
        )
        torch.save(
            {"input_ids": test_input_ids, "attention_mask": test_attention_mask},
            test_path,
        )
        test_blocks = int(test_input_ids.shape[0])
    else:
        if test_path.exists():
            test_path.unlink()
        LOGGER.warning(
            "Test-Split zu klein (%d Tokens); es wird keine test.pt geschrieben.",
            len(test_ids),
        )

    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "split_strategy": split_strategy,
        "raw_dir": str(Path(raw_dir)),
        "raw_files": [str(path) for path in files],
        "tokenizer_dir": str(Path(tokenizer_dir)),
        "tokenized_dir": str(tokenized_path),
        "block_size": block_size,
        "train_units": len(train_units),
        "validation_units": len(validation_units),
        "test_units": len(test_units),
        "train_tokens": len(train_ids),
        "validation_tokens": len(validation_ids),
        "test_tokens": len(test_ids),
        "train_blocks": int(train_input_ids.shape[0]),
        "validation_blocks": int(val_input_ids.shape[0]),
        "test_blocks": test_blocks,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    for output_file in [tokenized_path / "metadata.json", processed_path / "split_metadata.json"]:
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

    LOGGER.info(
        "Daten vorbereitet: %d Train-Bloecke, %d Validation-Bloecke, %d Test-Bloecke.",
        train_input_ids.shape[0],
        val_input_ids.shape[0],
        test_blocks,
    )
    return metadata


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenisiert Rohdaten fuer das Lumen-Smoke-Training."
    )
    parser.add_argument(
        "--config", default="configs/smoke_5m.yaml", help="Pfad zur YAML-Konfiguration."
    )
    parser.add_argument("--raw-dir", help="Ordner mit .txt-Rohdaten.")
    parser.add_argument("--tokenizer-dir", help="Ordner mit lokal trainiertem Tokenizer.")
    parser.add_argument("--tokenized-dir", help="Zielordner fuer train.pt und validation.pt.")
    parser.add_argument("--processed-dir", help="Zielordner fuer Split-Metadaten.")
    parser.add_argument("--validation-ratio", type=float, help="Anteil der Validierungsdaten.")
    parser.add_argument("--test-ratio", type=float, help="Anteil der Testdaten (Held-out).")
    parser.add_argument("--block-size", type=int, help="Sequenzlaenge in Tokens.")
    parser.add_argument("--seed", type=int, help="Seed fuer reproduzierbaren Split.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    config = load_config(args.config)
    data_config = config["data"]
    tokenizer_config = config["tokenizer"]

    prepare_data(
        raw_dir=args.raw_dir or data_config["raw_dir"],
        tokenizer_dir=args.tokenizer_dir or tokenizer_config["output_dir"],
        tokenized_dir=args.tokenized_dir or data_config["tokenized_dir"],
        processed_dir=args.processed_dir or data_config["processed_dir"],
        validation_ratio=(
            args.validation_ratio
            if args.validation_ratio is not None
            else float(data_config["validation_ratio"])
        ),
        test_ratio=(
            args.test_ratio
            if args.test_ratio is not None
            else float(data_config.get("test_ratio", 0.1))
        ),
        block_size=args.block_size or int(data_config["block_size"]),
        seed=args.seed if args.seed is not None else int(config["seed"]),
        special_tokens=tokenizer_config.get("special_tokens"),
    )


if __name__ == "__main__":
    main()
