"""Validiert die finale quantum-1 Daten- und Tokenisierungsbasis.

Das Skript prueft finale Pfade, Split-Trennung, Cleaning-Invarianten und
optional die tokenisierten 512er Sequenzen. Es startet kein Training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable

import torch
import yaml


SPLITS = ("train", "validation", "test")
REQUIRED_DATA_MANIFEST_FIELDS = [
    "dataset_name",
    "dataset_version",
    "source",
    "download_date_utc",
    "license_hint",
    "seed",
    "filter_rules",
    "document_counts",
    "text_amount",
    "estimated_tokens",
    "file_hashes",
]


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ungueltiges JSONL in {path}:{line_number}") from exc
    return records


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def path_parts(path: str | Path) -> tuple[str, ...]:
    return PurePosixPath(str(path).replace("\\", "/")).parts


def contains_sequence(path: str | Path, expected: tuple[str, ...]) -> bool:
    parts = path_parts(path)
    width = len(expected)
    return any(parts[index : index + width] == expected for index in range(len(parts) - width + 1))


def reject_pilot_path(path: str | Path, label: str) -> None:
    normalized = str(path).replace("\\", "/").lower()
    if "pilot" in normalized:
        raise ValueError(f"{label} darf keinen Pilotpfad verwenden: {path}")


def require_final_data_path(path: str | Path, label: str) -> None:
    reject_pilot_path(path, label)
    if not contains_sequence(path, ("data", "quantum", "final")):
        raise ValueError(f"{label} muss unter data/quantum/final liegen: {path}")


def require_final_tokenizer_path(path: str | Path, label: str) -> None:
    reject_pilot_path(path, label)
    if not contains_sequence(path, ("tokenizer", "quantum-1")):
        raise ValueError(f"{label} muss unter tokenizer/quantum-1 liegen: {path}")


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\wÄÖÜäöüß]+\b", text, flags=re.UNICODE))


def has_broken_text(text: str) -> bool:
    if "\ufffd" in text or "\x00" in text:
        return True
    control_chars = [char for char in text if ord(char) < 32 and char not in "\n\r\t"]
    return bool(control_chars)


def validate_config_paths(data_config: dict, tokenization_config: dict | None) -> None:
    for key in ["raw_dir", "cleaned_dir", "manifest_dir", "report_dir"]:
        require_final_data_path(data_config["paths"][key], f"paths.{key}")

    if tokenization_config is None:
        return
    require_final_tokenizer_path(tokenization_config["tokenizer"]["dir"], "tokenizer.dir")
    for key in ["train_file", "validation_file", "test_file"]:
        require_final_data_path(tokenization_config["input"][key], f"input.{key}")
    require_final_data_path(tokenization_config["output"]["dir"], "output.dir")


def validate_data_manifest(config: dict) -> dict:
    manifest_path = Path(config["paths"]["manifest_dir"]) / "data_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Finales Datenmanifest fehlt: {manifest_path}")
    manifest = read_json(manifest_path)
    missing = [field for field in REQUIRED_DATA_MANIFEST_FIELDS if field not in manifest]
    if missing:
        raise ValueError(f"Pflichtfelder fehlen im Datenmanifest: {missing}")
    if manifest.get("seed") != int(config["seed"]):
        raise ValueError("Seed im Datenmanifest stimmt nicht mit der finalen Datenconfig ueberein.")
    if manifest.get("source", {}).get("hf_dataset") != config["source"]["hf_dataset"]:
        raise ValueError("Quelle im Datenmanifest stimmt nicht mit der finalen Datenconfig ueberein.")
    return manifest


def validate_cleaned_split_records(config: dict) -> tuple[dict[str, dict], dict[str, set[str]]]:
    cleaned_dir = Path(config["paths"]["cleaned_dir"])
    rules = config["cleaning"]
    min_chars = int(rules["min_chars"])
    min_words = int(rules["min_words"])
    split_stats: dict[str, dict] = {}
    split_hashes: dict[str, set[str]] = {}
    seen_hashes: dict[str, str] = {}

    for split in SPLITS:
        path = cleaned_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Finaler {split}-Split fehlt: {path}")
        records = read_jsonl(path)
        hashes: set[str] = set()
        chars = 0
        words = 0
        for index, record in enumerate(records, start=1):
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Leerer Text in {path}:{index}")
            if len(text) < min_chars:
                raise ValueError(f"Zu kurzer Text in {path}:{index}")
            current_words = word_count(text)
            if current_words < min_words:
                raise ValueError(f"Zu wenige Woerter in {path}:{index}")
            if has_broken_text(text):
                raise ValueError(f"Kaputter Text in {path}:{index}")
            digest = str(record.get("sha256") or sha256_text(text))
            if digest != sha256_text(text):
                raise ValueError(f"sha256 passt nicht zum Text in {path}:{index}")
            if digest in hashes:
                raise ValueError(f"Exaktes Duplikat im Split {split}: {digest}")
            if digest in seen_hashes:
                raise ValueError(f"Split-Ueberschneidung: {digest} in {seen_hashes[digest]} und {split}")
            if record.get("split") not in (None, split):
                raise ValueError(f"Split-Feld falsch in {path}:{index}: {record.get('split')!r}")
            hashes.add(digest)
            seen_hashes[digest] = split
            chars += len(text)
            words += current_words
        split_hashes[split] = hashes
        split_stats[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "documents": len(records),
            "chars": chars,
            "words": words,
        }
    return split_stats, split_hashes


def tokenized_file_for_split(output_dir: str | Path, split: str) -> Path:
    return Path(output_dir) / f"{split}.pt"


def resolve_vocab_size(tokenization_config: dict, tokenization_manifest: dict) -> int:
    manifest_vocab = tokenization_manifest.get("tokenizer", {}).get("vocab_size")
    if manifest_vocab is not None:
        return int(manifest_vocab)
    tokenizer_dir = Path(tokenization_config["tokenizer"]["dir"])
    manifest_file = tokenizer_dir / tokenization_config["tokenizer"].get("manifest_file", "tokenizer_manifest.json")
    if manifest_file.exists():
        return int(read_json(manifest_file)["actual_vocab_size"])
    raise FileNotFoundError("vocab_size konnte weder aus Tokenisierungs- noch Tokenizer-Manifest gelesen werden.")


def validate_tokenized_splits(tokenization_config: dict) -> dict:
    output_dir = Path(tokenization_config["output"]["dir"])
    manifest_path = output_dir / tokenization_config["output"].get("manifest_file", "tokenization_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Finales Tokenisierungsmanifest fehlt: {manifest_path}")
    tokenization_manifest = read_json(manifest_path)
    context_length = int(tokenization_config["packing"]["context_length"])
    vocab_size = resolve_vocab_size(tokenization_config, tokenization_manifest)
    seen_hashes: dict[str, str] = {}
    split_reports: dict[str, dict] = {}

    for split in SPLITS:
        path = tokenized_file_for_split(output_dir, split)
        if not path.exists():
            raise FileNotFoundError(f"Tokenisierter finaler {split}-Split fehlt: {path}")
        data = torch.load(path, map_location="cpu", weights_only=True)
        required = {"input_ids", "attention_mask", "labels"}
        missing = required.difference(data)
        if missing:
            raise ValueError(f"{path} enthaelt nicht alle Pflichtfelder: {sorted(missing)}")
        input_ids = data["input_ids"].long()
        attention_mask = data["attention_mask"].long()
        labels = data["labels"].long()
        if input_ids.ndim != 2 or input_ids.shape[1] != context_length:
            raise ValueError(f"{path} muss Sequenzen mit exakt {context_length} Tokens enthalten.")
        if input_ids.shape != attention_mask.shape or input_ids.shape != labels.shape:
            raise ValueError(f"input_ids, attention_mask und labels haben unterschiedliche Formen in {path}.")
        if input_ids.numel() == 0:
            raise ValueError(f"{path} enthaelt keine Sequenzen.")
        minimum = int(input_ids.min().item())
        maximum = int(input_ids.max().item())
        if minimum < 0 or maximum >= vocab_size:
            raise ValueError(
                f"Ungueltige Token-ID in {path}: min={minimum}, max={maximum}, vocab_size={vocab_size}."
            )

        metadata = data.get("metadata", {})
        document_hashes = list(metadata.get("document_hashes") or [])
        if not document_hashes:
            raise ValueError(f"{path} muss document_hashes in metadata enthalten.")
        for digest in document_hashes:
            if digest in seen_hashes:
                raise ValueError(f"Tokenisierte Split-Ueberschneidung: {digest} in {seen_hashes[digest]} und {split}")
            seen_hashes[digest] = split

        actual_tokens = int(attention_mask.sum().item())
        metadata_tokens = int(metadata.get("tokens_before_padding", actual_tokens))
        if actual_tokens != metadata_tokens:
            raise ValueError(f"Tokenzaehlung in {path} stimmt nicht: mask={actual_tokens}, metadata={metadata_tokens}.")
        limit_key = f"{split}_max_tokens"
        limit = tokenization_config.get("limits", {}).get(limit_key)
        if limit is not None and metadata_tokens > int(limit):
            raise ValueError(f"{split} ueberschreitet Tokenlimit: {metadata_tokens} > {limit}")

        manifest_split = tokenization_manifest.get("splits", {}).get(split, {})
        if manifest_split and int(manifest_split.get("tokens_before_padding", metadata_tokens)) != metadata_tokens:
            raise ValueError(f"Tokenisierungsmanifest stimmt fuer {split} nicht mit .pt-Metadaten ueberein.")

        split_reports[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "sequence_count": int(input_ids.shape[0]),
            "context_length": int(input_ids.shape[1]),
            "tokens_before_padding": metadata_tokens,
            "tokens_including_padding": int(input_ids.numel()),
            "vocab_size": vocab_size,
            "document_hash_count": len(document_hashes),
        }
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "context_length": context_length,
        "vocab_size": vocab_size,
        "splits": split_reports,
    }


def validate_final_data(
    data_config_path: str | Path = "configs/quantum_1_final_data.yaml",
    tokenization_config_path: str | Path | None = None,
    require_tokenized: bool = False,
    skip_tokenized: bool = False,
    allow_missing: bool = False,
) -> dict:
    data_config = load_yaml(data_config_path)
    if skip_tokenized:
        tokenization_config = None
    elif tokenization_config_path:
        tokenization_config = load_yaml(tokenization_config_path)
    elif {"tokenizer", "input", "output", "packing"}.issubset(data_config):
        tokenization_config = data_config
    else:
        tokenization_config = None
    validate_config_paths(data_config, tokenization_config)

    if allow_missing:
        return {
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_config": str(data_config_path),
            "tokenization_config": str(tokenization_config_path) if tokenization_config_path else str(data_config_path),
            "source": data_config["source"],
            "seed": int(data_config["seed"]),
            "targets": data_config.get("targets", {}),
            "artifact_checks": "skipped",
            "ok": True,
        }

    manifest = validate_data_manifest(data_config)
    cleaned_stats, split_hashes = validate_cleaned_split_records(data_config)

    tokenized_report = None
    if tokenization_config is not None:
        output_dir = Path(tokenization_config["output"]["dir"])
        manifest_name = tokenization_config["output"].get("manifest_file", "tokenization_manifest.json")
        tokenized_exists = (output_dir / manifest_name).exists()
        if require_tokenized or tokenized_exists:
            tokenized_report = validate_tokenized_splits(tokenization_config)

    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_config": str(data_config_path),
        "tokenization_config": str(tokenization_config_path) if tokenization_config_path and not skip_tokenized else None,
        "source": data_config["source"],
        "seed": int(data_config["seed"]),
        "sampling_seed": int(data_config["sampling"]["split_seed"]),
        "targets": data_config.get("targets", {}),
        "cleaned_splits": cleaned_stats,
        "split_hash_counts": {split: len(hashes) for split, hashes in split_hashes.items()},
        "data_manifest_sha256": sha256_file(Path(data_config["paths"]["manifest_dir"]) / "data_manifest.json"),
        "manifest_dataset_version": manifest.get("dataset_version"),
        "tokenized": tokenized_report,
        "ok": True,
    }
    report_dir = Path(data_config["paths"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "final_data_validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validiert finale quantum-1 Daten und optional Tokenisierung.")
    parser.add_argument("--data-config", "--config", dest="data_config", default="configs/quantum_1_final_data.yaml")
    parser.add_argument("--tokenization-config", default=None)
    parser.add_argument("--require-tokenized", action="store_true", help="Tokenisierte .pt-Splits zwingend pruefen.")
    parser.add_argument("--skip-tokenized", action="store_true", help="Nur Roh/Clean/Split/Manifest pruefen.")
    parser.add_argument("--allow-missing", action="store_true", help="Nur Config/Pfade pruefen, fehlende Artefakte erlauben.")
    parser.add_argument("--json", action="store_true", help="Validierungsreport als JSON ausgeben.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = validate_final_data(
        data_config_path=args.data_config,
        tokenization_config_path=args.tokenization_config,
        require_tokenized=args.require_tokenized,
        skip_tokenized=args.skip_tokenized,
        allow_missing=args.allow_missing,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("Finale Datenbasis validiert.")


if __name__ == "__main__":
    main()
