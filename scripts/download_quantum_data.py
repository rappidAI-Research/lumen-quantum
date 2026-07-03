"""Streamt eine kleine Pilotmenge aus FineWeb2-HQ deu_Latn.

Kein voller Dataset-Download: Es wird Hugging Face Datasets im Streaming-Modus
verwendet und nach max_documents oder max_raw_bytes gestoppt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


LOGGER = logging.getLogger("lumen.download_quantum_data")


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


def nested_get(record: dict, dotted_key: str) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def first_text_field(record: dict, field_names: list[str]) -> tuple[str, str | None]:
    for field in field_names:
        value = nested_get(record, field)
        if isinstance(value, str) and value.strip():
            return value, field
    return "", None


def first_optional_field(record: dict, field_names: list[str]) -> str | None:
    for field in field_names:
        value = nested_get(record, field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def stable_doc_id(dataset: str, subset: str, text: str, source_url: str | None) -> str:
    payload = f"{dataset}\n{subset}\n{source_url or ''}\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def metadata_without_text(record: dict, text_field: str | None, max_value_chars: int = 500) -> dict:
    metadata: dict[str, Any] = {}
    for key, value in record.items():
        if key == text_field:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, str) and len(value) > max_value_chars:
                metadata[key] = value[:max_value_chars] + "...[truncated]"
            else:
                metadata[key] = value
    return metadata


def raw_record_from_hf(record: dict, config: dict, index: int) -> dict:
    source = config["source"]
    download = config["download"]
    text, text_field = first_text_field(record, download["text_fields"])
    url = first_optional_field(record, download.get("url_fields", []))
    doc_id = stable_doc_id(source["hf_dataset"], source["hf_subset"], text, url)
    raw_bytes = len(text.encode("utf-8"))
    return {
        "id": doc_id,
        "source_dataset": source["hf_dataset"],
        "source_subset": source["hf_subset"],
        "source_revision": source["revision"],
        "source_split": source["split"],
        "source_index": index,
        "source_url": url,
        "text": text,
        "text_field": text_field,
        "raw_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "raw_text_bytes": raw_bytes,
        "metadata": metadata_without_text(record, text_field),
    }


def iter_fineweb2_records(config: dict):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Das Paket 'datasets' fehlt. Installiere zuerst: pip install -r requirements.txt"
        ) from exc

    source = config["source"]
    download = config["download"]
    dataset = load_dataset(
        source["hf_dataset"],
        name=source["hf_subset"],
        split=source["split"],
        revision=source.get("revision", "main"),
        streaming=True,
    )
    dataset = dataset.shuffle(
        seed=int(config["seed"]),
        buffer_size=int(download.get("shuffle_buffer_size", 10000)),
    )
    return dataset


def write_jsonl(records: Iterable[dict], output_file: Path) -> int:
    count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def collect_pilot_records(config: dict) -> tuple[list[dict], dict]:
    max_documents = int(config["download"]["max_documents"])
    max_raw_bytes = int(config["download"]["max_raw_bytes"])
    records: list[dict] = []
    total_bytes = 0

    for index, hf_record in enumerate(iter_fineweb2_records(config)):
        raw_record = raw_record_from_hf(dict(hf_record), config, index)
        record_bytes = int(raw_record["raw_text_bytes"])
        if records and total_bytes + record_bytes > max_raw_bytes:
            break
        records.append(raw_record)
        total_bytes += record_bytes
        if len(records) >= max_documents or total_bytes >= max_raw_bytes:
            break

    stats = {
        "downloaded_documents": len(records),
        "raw_text_bytes": total_bytes,
        "max_documents": max_documents,
        "max_raw_bytes": max_raw_bytes,
    }
    return records, stats


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    raw_dir = Path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    records, stats = collect_pilot_records(config)
    if not records:
        raise ValueError("Keine FineWeb2-HQ-Dokumente geladen.")

    output_file = raw_dir / "fineweb2_hq_deu_latn_raw.jsonl"
    write_jsonl(records, output_file)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": config["source"],
        "seed": int(config["seed"]),
        "download": config["download"],
        "stats": stats,
        "output_file": str(output_file),
        "note": "Streaming-Pilotdownload; kein vollstaendiger Dataset-Download.",
    }
    (raw_dir / "download_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info(
        "%d FineWeb2-HQ Rohdokumente geschrieben (%d Bytes Text): %s",
        stats["downloaded_documents"],
        stats["raw_text_bytes"],
        output_file,
    )
    return output_file


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streamt eine FineWeb2-HQ deu_Latn Pilotmenge.")
    parser.add_argument("--config", default="configs/quantum_1_data.yaml")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()
