"""Erstellt einen kleinen Bericht ueber die quantum-1 Pilotdaten."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import yaml

try:
    from .clean_quantum_data import read_jsonl
except ImportError:
    from clean_quantum_data import read_jsonl


LOGGER = logging.getLogger("lumen.inspect_quantum_data")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def summarize_records(records: list[dict]) -> dict:
    lengths = sorted(len(record.get("text", "")) for record in records)
    source_counter = Counter(record.get("source_dataset", "unknown") for record in records)
    url_count = sum(1 for record in records if record.get("source_url"))
    if not records:
        return {"documents": 0}
    return {
        "documents": len(records),
        "chars": sum(lengths),
        "min_chars": lengths[0],
        "median_chars": lengths[len(lengths) // 2],
        "max_chars": lengths[-1],
        "approx_tokens": sum(
            int(record.get("approx_token_count", max(1, round(len(record.get("text", "")) / 4))))
            for record in records
        ),
        "documents_with_url": url_count,
        "source_counts": dict(source_counter),
        "sample_ids": [record.get("id") for record in records[:5]],
    }


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    cleaned_dir = Path(config["paths"]["cleaned_dir"])
    report_dir = Path(config["paths"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": config["source"],
        "seed": int(config["seed"]),
        "splits": {},
    }
    for name in ["documents_cleaned", "train", "validation", "test"]:
        path = cleaned_dir / f"{name}.jsonl"
        if path.exists():
            report["splits"][name] = summarize_records(read_jsonl(path))
        else:
            report["splits"][name] = {"documents": 0, "missing": True}

    output_file = report_dir / "quantum_data_report.json"
    output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Report geschrieben: %s", output_file)
    return output_file


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspiziert quantum-1 Pilotdaten.")
    parser.add_argument("--config", default="configs/quantum_1_data.yaml")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()
