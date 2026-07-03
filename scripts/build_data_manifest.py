"""Schreibt ein reproduzierbares Manifest fuer die quantum-1-Datenpipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

try:
    from .clean_quantum_data import read_jsonl
except ImportError:
    from clean_quantum_data import read_jsonl


LOGGER = logging.getLogger("lumen.build_data_manifest")


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
        return {"exists": False}
    records = read_jsonl(path)
    chars = sum(int(record.get("char_count", len(record.get("text", "")))) for record in records)
    words = sum(int(record.get("word_count", len(str(record.get("text", "")).split()))) for record in records)
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


def build_manifest(config_path: str | Path) -> dict:
    config = load_config(config_path)
    raw_dir = Path(config["paths"]["raw_dir"])
    cleaned_dir = Path(config["paths"]["cleaned_dir"])

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": config["manifest"]["version"],
        "dataset_name": config["project"]["dataset_name"],
        "seed": int(config["seed"]),
        "config_file": str(config_path),
        "config_sha256": file_sha256(config_path),
        "paths": config["paths"],
        "notes": config["manifest"].get("notes", ""),
        "download": load_json(raw_dir / "download_metadata.json"),
        "cleaning": load_json(cleaned_dir / "cleaning_metadata.json"),
        "splits": load_json(cleaned_dir / "split_metadata.json"),
        "files": {
            "raw": summarize_jsonl(raw_dir / "documents.jsonl"),
            "cleaned": summarize_jsonl(cleaned_dir / "documents_cleaned.jsonl"),
            "train": summarize_jsonl(cleaned_dir / "train.jsonl"),
            "validation": summarize_jsonl(cleaned_dir / "validation.jsonl"),
            "test": summarize_jsonl(cleaned_dir / "test.jsonl"),
        },
    }
    return manifest


def manifest_markdown(manifest: dict) -> str:
    files = manifest["files"]
    lines = [
        "# Data Manifest: quantum-1",
        "",
        f"- Dataset: `{manifest['dataset_name']}`",
        f"- Version: `{manifest['dataset_version']}`",
        f"- Created UTC: `{manifest['created_at_utc']}`",
        f"- Seed: `{manifest['seed']}`",
        f"- Notes: {manifest.get('notes', '')}",
        "",
        "## Files",
        "",
        "| Split | Documents | Approx Tokens | SHA256 |",
        "|---|---:|---:|---|",
    ]
    for name in ["raw", "cleaned", "train", "validation", "test"]:
        item = files[name]
        if item.get("exists"):
            lines.append(
                f"| {name} | {item['documents']} | {item['approx_tokens']} | `{item['sha256']}` |"
            )
        else:
            lines.append(f"| {name} | 0 | 0 | missing |")

    cleaning = manifest.get("cleaning", {})
    if cleaning:
        lines.extend(
            [
                "",
                "## Cleaning",
                "",
                f"- Input documents: `{cleaning.get('stats', {}).get('input_documents')}`",
                f"- Kept documents: `{cleaning.get('stats', {}).get('kept_documents')}`",
                f"- Rejected documents: `{cleaning.get('stats', {}).get('rejected_documents')}`",
                f"- Rules: `{json.dumps(cleaning.get('cleaning_rules', {}), ensure_ascii=False)}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "Run the pipeline from the repository root:",
            "",
            "```bash",
            "python scripts/download_quantum_data.py --config configs/quantum_1_data.yaml",
            "python scripts/clean_quantum_data.py --config configs/quantum_1_data.yaml",
            "python scripts/sample_quantum_data.py --config configs/quantum_1_data.yaml",
            "python scripts/build_data_manifest.py --config configs/quantum_1_data.yaml",
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
    parser = argparse.ArgumentParser(description="Erstellt ein quantum-1 Datenmanifest.")
    parser.add_argument("--config", default="configs/quantum_1_data.yaml")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()
