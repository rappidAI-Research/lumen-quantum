"""Entfernt Ueberschneidungen mit der bisherigen quantum-1-pilot-Datenbasis.

Laeuft NACH clean_quantum_data.py und VOR sample_quantum_data.py.

Vorgehen:
1. Baue eine Fingerprint-Menge aus dem bestehenden (alten) Datensatz. Es werden
   nur die konfigurierten Felder gelesen (Standard: normalisierter Text-SHA256
   'sha256' und stabile Dokument-ID 'id').
2. Filtere aus dem NEUEN cleaned-Datensatz jedes Dokument, dessen Fingerprint
   bereits im alten Datensatz vorkommt.
3. Schreibe den gefilterten Datensatz zurueck in den NEUEN cleaned-Pfad und einen
   ausfuehrlichen overlap_report.json.

Der alte Datensatz wird ausschliesslich gelesen und niemals veraendert.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import yaml

try:
    from .clean_quantum_data import read_jsonl, write_jsonl
except ImportError:
    from clean_quantum_data import read_jsonl, write_jsonl


LOGGER = logging.getLogger("lumen.exclude_known_documents")

DEFAULT_PREVIOUS_FILES = (
    "documents_cleaned.jsonl",
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
)
DEFAULT_FINGERPRINT_FIELDS = ("sha256", "id")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def record_fingerprints(record: dict, fields: Iterable[str]) -> set[str]:
    values: set[str] = set()
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            values.add(f"{field}:{value.strip()}")
    return values


def build_exclusion_fingerprints(
    previous_dirs: Iterable[str | Path],
    previous_files: Iterable[str],
    fields: Iterable[str],
) -> tuple[set[str], dict]:
    fields = list(fields)
    fingerprints: set[str] = set()
    sources: dict[str, int] = {}
    documents_scanned = 0
    for directory in previous_dirs:
        directory_path = Path(directory)
        for filename in previous_files:
            file_path = directory_path / filename
            if not file_path.exists():
                continue
            count = 0
            for record in read_jsonl(file_path):
                documents_scanned += 1
                before = len(fingerprints)
                fingerprints |= record_fingerprints(record, fields)
                count += 1
                _ = before
            sources[str(file_path)] = count
    return fingerprints, {
        "documents_scanned": documents_scanned,
        "sources": sources,
        "fields": fields,
    }


def filter_new_documents(
    new_records: list[dict],
    exclusion: set[str],
    fields: Iterable[str],
) -> tuple[list[dict], list[dict]]:
    fields = list(fields)
    kept: list[dict] = []
    removed: list[dict] = []
    for record in new_records:
        overlap = record_fingerprints(record, fields) & exclusion
        if overlap:
            removed.append(record)
        else:
            kept.append(record)
    return kept, removed


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    overlap_config = config.get("overlap")
    if not overlap_config:
        raise ValueError("Config enthaelt keinen 'overlap'-Abschnitt.")

    cleaned_dir = Path(config["paths"]["cleaned_dir"])
    cleaned_file = cleaned_dir / "documents_cleaned.jsonl"
    if not cleaned_file.exists():
        raise FileNotFoundError(
            f"Neuer bereinigter Datensatz fehlt: {cleaned_file}. Fuehre zuerst clean_quantum_data.py aus."
        )

    previous_dirs = [Path(p) for p in overlap_config.get("previous_dataset_dirs", [])]
    previous_files = overlap_config.get("previous_dataset_files", list(DEFAULT_PREVIOUS_FILES))
    fields = overlap_config.get("fingerprint_fields", list(DEFAULT_FINGERPRINT_FIELDS))

    # Sicherheit: niemals in den alten Datensatz schreiben.
    for directory in previous_dirs:
        if directory.resolve() == cleaned_dir.resolve():
            raise ValueError(
                f"overlap.previous_dataset_dirs darf nicht das neue cleaned_dir sein: {cleaned_dir}."
            )

    exclusion, exclusion_stats = build_exclusion_fingerprints(previous_dirs, previous_files, fields)
    LOGGER.info(
        "Fingerprints aus altem Datensatz: %d (aus %d Dokumenten).",
        len(exclusion),
        exclusion_stats["documents_scanned"],
    )

    new_records = read_jsonl(cleaned_file)
    kept, removed = filter_new_documents(new_records, exclusion, fields)

    if not kept:
        raise ValueError(
            "Nach der Overlap-Filterung sind keine Dokumente uebrig. "
            "Pruefe die Fingerprint-Felder und den alten Datensatz."
        )

    # Nur den NEUEN Datensatz zurueckschreiben.
    write_jsonl(kept, cleaned_file)

    report_file = Path(
        overlap_config.get("report_file", cleaned_dir.parent / "reports" / "overlap_report.json")
    )
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_file": str(config_path),
        "new_cleaned_file": str(cleaned_file),
        "previous_dataset_dirs": [str(p) for p in previous_dirs],
        "previous_dataset_files": list(previous_files),
        "fingerprint_fields": list(fields),
        "old_dataset_scan": exclusion_stats,
        "old_dataset_unique_fingerprints": len(exclusion),
        "new_documents_before": len(new_records),
        "new_documents_removed_as_overlap": len(removed),
        "new_documents_kept": len(kept),
        "overlap_ratio": round(len(removed) / max(1, len(new_records)), 6),
        "known_limitations": overlap_config.get(
            "known_limitations",
            "Nur exakte Fingerprints werden verglichen; Near-Duplicates werden nicht erkannt.",
        ),
        "sample_removed_ids": [r.get("id") for r in removed[:20]],
    }
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info(
        "Overlap-Filterung: %d/%d Dokumente behalten, %d als Ueberschneidung entfernt. Report: %s",
        len(kept),
        len(new_records),
        len(removed),
        report_file,
    )
    return report_file


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entfernt Ueberschneidungen mit der bisherigen quantum-1-pilot-Datenbasis."
    )
    parser.add_argument("--config", default="configs/quantum_1_6_pilot_data.yaml")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()
