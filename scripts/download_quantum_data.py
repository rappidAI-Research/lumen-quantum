"""Erzeugt oder laedt eine kleine Rohdaten-Stichprobe fuer quantum-1.

Standardmaessig arbeitet dieses Skript offline und schreibt eine kleine,
reproduzierbare deutschsprachige Startstichprobe. Netzwerkdownloads sind nur
aktiv, wenn --allow-network gesetzt wird und Quellen in der YAML stehen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml


LOGGER = logging.getLogger("lumen.download_quantum_data")


SEED_DOCUMENTS = [
    {
        "source_id": "lumen_seed_de_v0",
        "title": "Alltag und Sprache",
        "license": "synthetic-local",
        "text": (
            "Lumen ist ein lokaler deutschsprachiger Assistent. Er soll kurze Fragen "
            "verstehen, einfache Antworten geben und Texte in ruhiger Sprache erklaeren. "
            "Im Alltag helfen klare Saetze, gute Beispiele und ein vorsichtiger Umgang "
            "mit unsicheren Informationen."
        ),
    },
    {
        "source_id": "lumen_seed_de_v0",
        "title": "Technische Notizen",
        "license": "synthetic-local",
        "text": (
            "Ein kleines Sprachmodell wird zuerst mit einer ueberschaubaren Pipeline "
            "geprueft. Dazu gehoeren Rohdaten, Bereinigung, ein reproduzierbarer Split "
            "und ein Manifest. Erst wenn diese Schritte stabil laufen, lohnt sich ein "
            "groesseres Training mit mehr Parametern."
        ),
    },
    {
        "source_id": "lumen_seed_de_v0",
        "title": "Lernen und Schule",
        "license": "synthetic-local",
        "text": (
            "In der Schule lesen Kinder Texte, stellen Fragen und vergleichen Antworten. "
            "Ein deutsches Trainingsdataset sollte verschiedene Themen enthalten: Alltag, "
            "Technik, Natur, Lernen und einfache Erklaerungen. Doppelte Texte werden "
            "entfernt, damit die Auswertung ehrlich bleibt."
        ),
    },
    {
        "source_id": "lumen_seed_de_v0",
        "title": "Naturbeschreibung",
        "license": "synthetic-local",
        "text": (
            "Am Morgen liegt Nebel ueber der Wiese. Die Luft ist kuehl, und auf den "
            "Blaettern sammeln sich kleine Tropfen. Solche Beschreibungen helfen einem "
            "Modell, deutsche Satzmuster, Zeitangaben und einfache Beobachtungen zu "
            "lernen."
        ),
    },
    {
        "source_id": "lumen_seed_de_v0",
        "title": "Datenqualitaet",
        "license": "synthetic-local",
        "text": (
            "Gute Daten sind wichtiger als eine grosse Menge schlechter Daten. Sehr kurze "
            "Texte, kaputte Zeichenfolgen, reine Menues und doppelte Abschnitte werden "
            "gefiltert. Jede Version des Datensatzes braucht dokumentierte Quellen, Seeds "
            "und Filterregeln."
        ),
    },
    {
        "source_id": "lumen_seed_de_v0",
        "title": "Lokales Arbeiten",
        "license": "synthetic-local",
        "text": (
            "Das Projekt arbeitet lokal und laedt keine Modellgewichte. Fuer die "
            "Datenpipeline werden zunaechst nur kleine Textdateien vorbereitet. Spaeter "
            "koennen erlaubte deutschsprachige Quellen ergaenzt werden, wenn Lizenz, "
            "Version und Herkunft sauber dokumentiert sind."
        ),
    },
]


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


def stable_id(source_id: str, text: str) -> str:
    digest = hashlib.sha256(f"{source_id}\n{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def normalize_downloaded_text(text: str) -> str:
    text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def record_for_text(source_id: str, title: str, license_name: str, text: str, url: str | None = None) -> dict:
    normalized = text.strip()
    doc_id = stable_id(source_id, normalized)
    return {
        "id": doc_id,
        "source_id": source_id,
        "title": title,
        "license": license_name,
        "url": url,
        "text": normalized,
        "sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_jsonl(records: list[dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def download_url(url: str, timeout_seconds: int = 30) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "LumenQuantumDataSmoke/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def collect_records(config: dict, allow_network: bool) -> list[dict]:
    download_config = config.get("download", {})
    records: list[dict] = []

    if download_config.get("write_seed_sample", True):
        seed = int(config.get("seed", 0))
        rng = random.Random(seed)
        seed_docs = list(SEED_DOCUMENTS)
        rng.shuffle(seed_docs)
        for doc in seed_docs:
            records.append(
                record_for_text(
                    source_id=doc["source_id"],
                    title=doc["title"],
                    license_name=doc["license"],
                    text=doc["text"],
                )
            )

    sources = download_config.get("sources") or []
    if sources and not allow_network:
        LOGGER.warning("Quellen in der Config vorhanden, aber Netzwerkdownload ist deaktiviert.")

    if allow_network:
        max_per_source = int(download_config.get("max_documents_per_source", 20))
        for source in sources:
            source_id = source["id"]
            license_name = source.get("license", "unknown")
            urls = list(source.get("urls") or [])[:max_per_source]
            for url in urls:
                LOGGER.info("Lade Quelle %s: %s", source_id, url)
                text = normalize_downloaded_text(download_url(url))
                if text:
                    records.append(
                        record_for_text(
                            source_id=source_id,
                            title=source.get("name", source_id),
                            license_name=license_name,
                            text=text,
                            url=url,
                        )
                    )

    return records


def write_metadata(config: dict, records: list[dict], output_dir: Path, allow_network: bool) -> None:
    source_counts: dict[str, int] = {}
    for record in records:
        source_counts[record["source_id"]] = source_counts.get(record["source_id"], 0) + 1

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(config.get("seed", 0)),
        "allow_network": allow_network,
        "document_count": len(records),
        "source_counts": source_counts,
        "output_file": str(output_dir / "documents.jsonl"),
        "notes": "Rohdaten vor Cleaning; noch nicht fuer Training freigegeben.",
    }
    (output_dir / "download_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(config_path: str | Path, allow_network: bool = False) -> Path:
    config = load_config(config_path)
    raw_dir = Path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(config, allow_network=allow_network)
    if not records:
        raise ValueError("Keine Rohdaten erzeugt oder geladen.")

    output_file = raw_dir / "documents.jsonl"
    write_jsonl(records, output_file)
    write_metadata(config, records, raw_dir, allow_network)
    LOGGER.info("%d Rohdokumente geschrieben: %s", len(records), output_file)
    return output_file


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bereitet kleine quantum-1-Rohdaten vor.")
    parser.add_argument("--config", default="configs/quantum_1_data.yaml")
    parser.add_argument("--allow-network", action="store_true", help="Optionale URLs aus der Config laden.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    run(args.config, allow_network=args.allow_network)


if __name__ == "__main__":
    main()
