"""Erstellt reproduzierbare Train/Validation/Test-Splits fuer quantum-1-Daten."""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

try:
    from .clean_quantum_data import read_jsonl, write_jsonl
except ImportError:
    from clean_quantum_data import read_jsonl, write_jsonl


LOGGER = logging.getLogger("lumen.sample_quantum_data")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split-Ratios muessen 1.0 ergeben, aktuell: {total}")
    if min(train_ratio, validation_ratio, test_ratio) <= 0:
        raise ValueError("Alle Split-Ratios muessen groesser als 0 sein.")


def split_records(records: list[dict], seed: int, sampling_config: dict) -> dict[str, list[dict]]:
    max_documents = int(sampling_config["max_documents"])
    train_ratio = float(sampling_config["train_ratio"])
    validation_ratio = float(sampling_config["validation_ratio"])
    test_ratio = float(sampling_config["test_ratio"])
    validate_ratios(train_ratio, validation_ratio, test_ratio)

    by_hash: dict[str, dict] = {}
    for record in records:
        by_hash.setdefault(record["sha256"], record)
    unique_records = list(by_hash.values())

    rng = random.Random(seed)
    rng.shuffle(unique_records)
    selected = unique_records[:max_documents]
    n = len(selected)
    if n < 3:
        raise ValueError("Mindestens 3 bereinigte Dokumente sind fuer Train/Validation/Test noetig.")

    validation_count = max(1, round(n * validation_ratio))
    test_count = max(1, round(n * test_ratio))
    while n - validation_count - test_count < 1:
        if test_count > 1:
            test_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            raise ValueError("Zu wenige Dokumente fuer disjunkte Splits.")

    test = selected[:test_count]
    validation = selected[test_count : test_count + validation_count]
    train = selected[test_count + validation_count :]
    return {"train": train, "validation": validation, "test": test}


def assert_disjoint(splits: dict[str, list[dict]]) -> None:
    seen: dict[str, str] = {}
    for split_name, records in splits.items():
        for record in records:
            digest = record["sha256"]
            if digest in seen:
                raise ValueError(
                    f"Datenleck: Dokument {record.get('id')} in {split_name} und {seen[digest]}."
                )
            seen[digest] = split_name


def split_stats(splits: dict[str, list[dict]]) -> dict:
    stats = {}
    for split_name, records in splits.items():
        stats[split_name] = {
            "documents": len(records),
            "chars": sum(int(record.get("char_count", len(record.get("text", "")))) for record in records),
            "words": sum(int(record.get("word_count", 0)) for record in records),
            "approx_tokens": sum(int(record.get("approx_token_count", 0)) for record in records),
        }
    return stats


def run(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    cleaned_file = Path(config["paths"]["cleaned_dir"]) / "documents_cleaned.jsonl"
    if not cleaned_file.exists():
        raise FileNotFoundError(f"Bereinigte Daten fehlen: {cleaned_file}. Fuehre clean_quantum_data.py aus.")

    records = read_jsonl(cleaned_file)
    splits = split_records(records, int(config["seed"]), config["sampling"])
    assert_disjoint(splits)

    cleaned_dir = Path(config["paths"]["cleaned_dir"])
    output_paths: dict[str, Path] = {}
    for split_name, split_records_ in splits.items():
        output_path = cleaned_dir / f"{split_name}.jsonl"
        write_jsonl(split_records_, output_path)
        output_paths[split_name] = output_path

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(config["seed"]),
        "input_file": str(cleaned_file),
        "sampling": config["sampling"],
        "stats": split_stats(splits),
        "disjoint_by": "sha256",
    }
    (cleaned_dir / "split_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info(
        "Splits geschrieben: train=%d validation=%d test=%d",
        len(splits["train"]),
        len(splits["validation"]),
        len(splits["test"]),
    )
    return output_paths


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Erstellt quantum-1 Train/Validation/Test-Splits.")
    parser.add_argument("--config", default="configs/quantum_1_data.yaml")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()
