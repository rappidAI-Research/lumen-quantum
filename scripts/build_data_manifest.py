"""Schreibt ein Manifest fuer die FineWeb2-HQ quantum-1 Pilotdaten."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import yaml

try:
    from .clean_quantum_data import read_jsonl
except ImportError:
    from clean_quantum_data import read_jsonl


LOGGER = logging.getLogger("lumen.build_data_manifest")

REQUIRED_MANIFEST_FIELDS = [
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


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def file_sha256(path: str | Path) -> str | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_jsonl(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "documents": 0, "chars": 0, "approx_tokens": 0}
    records = read_jsonl(path)
    chars = sum(int(record.get("char_count", len(record.get("text", "")))) for record in records)
    words = sum(
        int(record.get("word_count", len(str(record.get("text", "")).split())))
        for record in records
    )
    approx_tokens = sum(
        int(record.get("approx_token_count", max(1, round(len(record.get("text", "")) / 4))))
        for record in records
    )
    return {
        "exists": True,
        "path": str(path),
        "sha256": file_sha256(path),
        "documents": len(records),
        "chars": chars,
        "words": words,
        "approx_tokens": approx_tokens,
    }


def assert_required_fields(manifest: dict) -> None:
    missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in manifest]
    if missing:
        raise ValueError(f"Manifest-Pflichtfelder fehlen: {missing}")


def build_manifest(config_path: str | Path) -> dict:
    config = load_config(config_path)
    raw_dir = Path(config["paths"]["raw_dir"])
    cleaned_dir = Path(config["paths"]["cleaned_dir"])
    report_dir = Path(config["paths"]["report_dir"])
    files = {
        "raw": summarize_jsonl(raw_dir / "fineweb2_hq_deu_latn_raw.jsonl"),
        "cleaned": summarize_jsonl(cleaned_dir / "documents_cleaned.jsonl"),
        "train": summarize_jsonl(cleaned_dir / "train.jsonl"),
        "validation": summarize_jsonl(cleaned_dir / "validation.jsonl"),
        "test": summarize_jsonl(cleaned_dir / "test.jsonl"),
        "inspection_report": {
            "exists": (report_dir / "quantum_data_report.json").exists(),
            "path": str(report_dir / "quantum_data_report.json"),
            "sha256": file_sha256(report_dir / "quantum_data_report.json"),
        },
    }
    download_metadata = load_json(raw_dir / "download_metadata.json")
    cleaning_metadata = load_json(cleaned_dir / "cleaning_metadata.json")
    split_metadata = load_json(cleaned_dir / "split_metadata.json")

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_file": str(config_path),
        "dataset_name": config["project"]["dataset_name"],
        "dataset_version": config["manifest"]["version"],
        "project_description": config["project"].get("description", ""),
        "source": config["source"],
        "download_date_utc": download_metadata.get("created_at_utc"),
        "license_hint": config["source"].get("license"),
        "seed": int(config["seed"]),
        "sampling_seed": int(config["sampling"]["split_seed"]),
        "targets": config.get("targets", {}),
        "filter_rules": config["cleaning"],
        "document_counts": {
            name: files[name]["documents"]
            for name in ["raw", "cleaned", "train", "validation", "test"]
        },
        "text_amount": {
            name: {
                "chars": files[name]["chars"],
                "words": files[name].get("words", 0),
            }
            for name in ["raw", "cleaned", "train", "validation", "test"]
        },
        "estimated_tokens": {
            name: files[name]["approx_tokens"]
            for name in ["raw", "cleaned", "train", "validation", "test"]
        },
        "file_hashes": {name: files[name].get("sha256") for name in files},
        "files": files,
        "download": download_metadata,
        "cleaning": cleaning_metadata,
        "splits": split_metadata,
        "notes": config["manifest"].get("notes", ""),
    }
    assert_required_fields(manifest)
    return manifest


def manifest_markdown(manifest: dict) -> str:
    config_file = manifest.get("config_file", "configs/quantum_1_data.yaml")
    title_suffix = "Final" if "final" in manifest.get("dataset_version", "").lower() else "Pilot"
    lines = [
        f"# Data Manifest: quantum-1 FineWeb2-HQ {title_suffix}",
        "",
        f"- Dataset: `{manifest['dataset_name']}`",
        f"- Version: `{manifest['dataset_version']}`",
        f"- Source: `{manifest['source']['hf_dataset']}` / `{manifest['source']['hf_subset']}`",
        f"- Revision: `{manifest['source']['revision']}`",
        f"- Config: `{config_file}`",
        f"- Download UTC: `{manifest['download_date_utc']}`",
        f"- Seed: `{manifest['seed']}`",
        f"- Split seed: `{manifest['sampling_seed']}`",
        f"- License hint: {manifest['license_hint']}",
        f"- Notes: {manifest.get('notes', '')}",
        "",
        "## Files",
        "",
        "| Split | Documents | Chars | Approx Tokens | SHA256 |",
        "|---|---:|---:|---:|---|",
    ]
    for name in ["raw", "cleaned", "train", "validation", "test"]:
        lines.append(
            f"| {name} | {manifest['document_counts'][name]} | "
            f"{manifest['text_amount'][name]['chars']} | {manifest['estimated_tokens'][name]} | "
            f"`{manifest['file_hashes'][name]}` |"
        )

    lines.extend(
        [
            "",
            "## Cleaning",
            "",
            f"- Rules: `{json.dumps(manifest['filter_rules'], ensure_ascii=False)}`",
            f"- Rejections: `{json.dumps(manifest.get('cleaning', {}).get('stats', {}).get('rejection_counts', {}), ensure_ascii=False)}`",
            "",
            "## Split",
            "",
            "- Method: stable SHA256 bucket, not random order after download.",
            "- Disjoint key: document text SHA256.",
            "- Target ratios: train 98 %, validation 1 %, test 1 %.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            f"python scripts/download_quantum_data.py --config {config_file}",
            f"python scripts/clean_quantum_data.py --config {config_file}",
            f"python scripts/sample_quantum_data.py --config {config_file}",
            f"python scripts/inspect_quantum_data.py --config {config_file}",
            f"python scripts/build_data_manifest.py --config {config_file}",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def run(config_path: str | Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    manifest_dir = Path(config["paths"]["manifest_dir"])
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(config_path)

    json_path = manifest_dir / "data_manifest.json"
    md_path = manifest_dir / "data_manifest.md"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(manifest_markdown(manifest), encoding="utf-8")
    LOGGER.info("Manifest geschrieben: %s und %s", json_path, md_path)
    return json_path, md_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erstellt das quantum-1 FineWeb2-HQ Datenmanifest."
    )
    parser.add_argument("--config", default="configs/quantum_1_data.yaml")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()
