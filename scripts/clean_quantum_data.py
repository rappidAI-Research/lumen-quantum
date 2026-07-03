"""Bereinigt quantum-1-Rohdaten und entfernt unbrauchbare Dokumente."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml


LOGGER = logging.getLogger("lumen.clean_quantum_data")

GERMAN_STOPWORDS = {
    "der", "die", "das", "und", "ist", "ein", "eine", "mit", "fuer", "nicht",
    "werden", "wird", "im", "in", "den", "dem", "zu", "auf", "von", "sich",
    "dass", "auch", "als", "oder", "wenn", "diese", "dieser", "diesen",
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


def read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Ungueltiges JSONL in {path}:{line_number}") from exc
    return records


def write_jsonl(records: list[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\wÄÖÜäöüß]+\b", text, flags=re.UNICODE))


def approx_token_count(text: str) -> int:
    # Grobe, tokenizerfreie Schaetzung fuer Manifest/Monitoring.
    return max(1, round(len(text) / 4))


def repeated_char_run_too_long(text: str, max_run: int) -> bool:
    if max_run <= 0:
        return False
    return re.search(rf"(.)\1{{{max_run},}}", text) is not None


def non_letter_ratio(text: str) -> float:
    non_space = [char for char in text if not char.isspace()]
    if not non_space:
        return 1.0
    letters = sum(1 for char in non_space if char.isalpha())
    return 1.0 - (letters / len(non_space))


def german_stopword_hits(text: str) -> int:
    words = re.findall(r"\b[\wÄÖÜäöüß]+\b", text.lower(), flags=re.UNICODE)
    return sum(1 for word in words if word in GERMAN_STOPWORDS)


def rejection_reasons(text: str, rules: dict) -> list[str]:
    reasons: list[str] = []
    chars = len(text)
    words = word_count(text)
    if chars < int(rules["min_chars"]):
        reasons.append("too_short_chars")
    if chars > int(rules["max_chars"]):
        reasons.append("too_long_chars")
    if words < int(rules["min_words"]):
        reasons.append("too_few_words")
    if repeated_char_run_too_long(text, int(rules["max_repeated_char_run"])):
        reasons.append("repeated_character_run")
    if non_letter_ratio(text) > float(rules["max_non_letter_ratio"]):
        reasons.append("too_many_non_letters")
    if german_stopword_hits(text) < int(rules["min_german_stopword_hits"]):
        reasons.append("not_enough_german_stopwords")

    lowered = text.lower()
    for pattern in rules.get("reject_patterns", []):
        if pattern.lower() in lowered:
            reasons.append(f"reject_pattern:{pattern}")
    return reasons


def clean_records(records: list[dict], rules: dict) -> tuple[list[dict], dict]:
    cleaned: list[dict] = []
    seen_hashes: set[str] = set()
    rejection_counts: Counter[str] = Counter()

    for record in records:
        text = normalize_text(str(record.get("text", "")))
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if rules.get("remove_exact_duplicates", True) and text_hash in seen_hashes:
            rejection_counts["exact_duplicate"] += 1
            continue
        seen_hashes.add(text_hash)

        reasons = rejection_reasons(text, rules)
        if reasons:
            rejection_counts.update(reasons)
            continue

        output = dict(record)
        output["text"] = text
        output["sha256"] = text_hash
        output["char_count"] = len(text)
        output["word_count"] = word_count(text)
        output["approx_token_count"] = approx_token_count(text)
        output["cleaned_at_utc"] = datetime.now(timezone.utc).isoformat()
        cleaned.append(output)

    stats = {
        "input_documents": len(records),
        "kept_documents": len(cleaned),
        "rejected_documents": len(records) - len(cleaned),
        "rejection_counts": dict(rejection_counts),
        "total_chars": sum(record["char_count"] for record in cleaned),
        "total_words": sum(record["word_count"] for record in cleaned),
        "approx_tokens": sum(record["approx_token_count"] for record in cleaned),
    }
    return cleaned, stats


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    raw_file = Path(config["paths"]["raw_dir"]) / "documents.jsonl"
    if not raw_file.exists():
        raise FileNotFoundError(f"Rohdaten fehlen: {raw_file}. Fuehre zuerst download_quantum_data.py aus.")

    cleaned_dir = Path(config["paths"]["cleaned_dir"])
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(raw_file)
    cleaned, stats = clean_records(records, config["cleaning"])
    if not cleaned:
        raise ValueError("Cleaning hat alle Dokumente entfernt. Filterregeln oder Rohdaten pruefen.")

    output_file = cleaned_dir / "documents_cleaned.jsonl"
    write_jsonl(cleaned, output_file)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(config.get("seed", 0)),
        "input_file": str(raw_file),
        "output_file": str(output_file),
        "cleaning_rules": config["cleaning"],
        "stats": stats,
    }
    (cleaned_dir / "cleaning_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("%d/%d Dokumente behalten: %s", stats["kept_documents"], stats["input_documents"], output_file)
    return output_file


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bereinigt quantum-1-Rohdaten.")
    parser.add_argument("--config", default="configs/quantum_1_data.yaml")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()
