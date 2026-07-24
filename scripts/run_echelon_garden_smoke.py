#!/usr/bin/env python3

import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sentencepiece as spm
import yaml
from datasets import load_dataset
from echelon_quality_filters import (
    SimHashIndex,
    alpha_ratio,
    boilerplate_match_count,
    duplicate_line_ratio,
    metadata_rejection,
    repeated_ngram_ratio,
    simhash64,
    words,
)

CONFIG_PATH = Path("configs/echelon/garden_smoke.yaml")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
EXCESSIVE_BLANK_LINES = re.compile(r"\n{3,}")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS.sub("", text)
    text = EXCESSIVE_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def ratio(count: int, total: int) -> float:
    return count / max(total, 1)


def quality_rejection(text: str, config: dict) -> str | None:
    filters = config["filters"]
    length = len(text)

    if length < filters["minimum_characters"]:
        return "too_short"

    if length > filters["maximum_characters"]:
        return "too_long"

    url_characters = sum(len(match.group(0)) for match in URL_PATTERN.finditer(text))
    if ratio(url_characters, length) > filters["maximum_url_ratio"]:
        return "url_ratio"

    digit_count = sum(character.isdigit() for character in text)
    if ratio(digit_count, length) > filters["maximum_digit_ratio"]:
        return "digit_ratio"

    symbol_count = sum(not character.isalnum() and not character.isspace() for character in text)
    if ratio(symbol_count, length) > filters["maximum_symbol_ratio"]:
        return "symbol_ratio"

    if len(words(text)) < filters["minimum_words"]:
        return "too_few_words"

    if alpha_ratio(text) < filters["minimum_alpha_ratio"]:
        return "alpha_ratio"

    if duplicate_line_ratio(text) > filters["maximum_duplicate_line_ratio"]:
        return "duplicate_lines"

    if repeated_ngram_ratio(text, n=5) > filters["maximum_repeated_5gram_ratio"]:
        return "repeated_5grams"

    if boilerplate_match_count(text) > filters["maximum_boilerplate_matches"]:
        return "boilerplate"

    return None


def normalized_fingerprint(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_split(fingerprint: str, config: dict) -> str:
    splits = config["splits"]
    bucket = int(fingerprint[:16], 16) % splits["total_buckets"]

    train_limit = splits["train_buckets"]
    validation_limit = train_limit + splits["validation_buckets"]

    if bucket < train_limit:
        return "train"

    if bucket < validation_limit:
        return "validation"

    return "test"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    config = load_config()
    output_root = Path(config["output"]["root"])
    cleaned_dir = output_root / "cleaned"
    tokenized_dir = output_root / "tokenized"
    manifest_path = Path(config["output"]["manifest"])

    cleaned_dir.mkdir(parents=True, exist_ok=True)
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = spm.SentencePieceProcessor(model_file=config["tokenizer"]["model"])

    if tokenizer.get_piece_size() > 65536:
        raise ValueError("Vokabular ist zu groß für uint16.")

    dataset = load_dataset(
        config["source"]["dataset"],
        config["source"]["configuration"],
        split=config["source"]["split"],
        streaming=config["source"]["streaming"],
    ).shuffle(
        seed=config["seed"],
        buffer_size=1000,
    )

    splits = ("train", "validation", "test")
    text_paths = {split: cleaned_dir / f"{split}.jsonl" for split in splits}
    token_paths = {split: tokenized_dir / f"{split}.bin" for split in splits}

    text_handles = {split: text_paths[split].open("w", encoding="utf-8") for split in splits}
    token_handles = {split: token_paths[split].open("wb") for split in splits}

    counters = Counter()
    documents_per_split = Counter()
    tokens_per_split = Counter()
    seen_fingerprints: set[str] = set()

    near_config = config["near_duplicate"]
    near_duplicate_index = SimHashIndex(
        bits=near_config["simhash_bits"],
        bands=near_config["bands"],
        maximum_hamming_distance=near_config["maximum_hamming_distance"],
    )

    try:
        for sample in dataset:
            counters["documents_seen"] += 1

            if counters["documents_seen"] > config["source"]["maximum_documents_seen"]:
                break

            metadata_reason = metadata_rejection(sample, config)

            if metadata_reason is not None:
                counters[f"rejected_{metadata_reason}"] += 1
                continue

            text = clean_text(sample.get("text", ""))
            rejection = quality_rejection(text, config)

            if rejection is not None:
                counters[f"rejected_{rejection}"] += 1
                continue

            fingerprint = normalized_fingerprint(text)

            if fingerprint in seen_fingerprints:
                counters["rejected_exact_duplicate"] += 1
                continue

            similarity_hash = simhash64(text)

            if near_config["enabled"] and near_duplicate_index.is_near_duplicate(similarity_hash):
                counters["rejected_near_duplicate"] += 1
                continue

            seen_fingerprints.add(fingerprint)

            if near_config["enabled"]:
                near_duplicate_index.add(similarity_hash)

            split = stable_split(fingerprint, config)

            record = {
                "fingerprint": fingerprint,
                "text": text,
            }
            text_handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")

            token_ids = tokenizer.encode(
                text,
                out_type=int,
                add_bos=False,
                add_eos=True,
            )

            np.asarray(token_ids, dtype=np.uint16).tofile(token_handles[split])

            documents_per_split[split] += 1
            tokens_per_split[split] += len(token_ids)
            counters["documents_accepted"] += 1

    finally:
        for handle in text_handles.values():
            handle.close()
        for handle in token_handles.values():
            handle.close()

    file_hashes = {}
    file_sizes = {}

    for split in splits:
        file_hashes[f"{split}_jsonl"] = sha256_file(text_paths[split])
        file_hashes[f"{split}_bin"] = sha256_file(token_paths[split])
        file_sizes[f"{split}_jsonl_bytes"] = text_paths[split].stat().st_size
        file_sizes[f"{split}_bin_bytes"] = token_paths[split].stat().st_size

    manifest = {
        "pipeline": "quantum-1-echelon-garden-smoke",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "counters": dict(counters),
        "documents_per_split": {split: documents_per_split[split] for split in splits},
        "tokens_per_split": {split: tokens_per_split[split] for split in splits},
        "total_tokens": sum(tokens_per_split.values()),
        "unique_fingerprints": len(seen_fingerprints),
        "near_duplicate_index_size": len(near_duplicate_index),
        "file_hashes": file_hashes,
        "file_sizes": file_sizes,
        "tokenizer_vocab_size": tokenizer.get_piece_size(),
        "token_dtype": "uint16",
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== GARDEN-SMOKE ABGESCHLOSSEN ===")
    print("Gesehene Dokumente:", counters["documents_seen"])
    print("Akzeptierte Dokumente:", counters["documents_accepted"])
    print("Dokumente je Split:", dict(documents_per_split))
    print("Tokens je Split:", dict(tokens_per_split))
    print("Tokens gesamt:", sum(tokens_per_split.values()))
    print("Manifest:", manifest_path)

    if counters["documents_accepted"] == 0:
        raise SystemExit("FEHLER: Kein Dokument hat die Filter bestanden.")


if __name__ == "__main__":
    main()
