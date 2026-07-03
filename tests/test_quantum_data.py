import json
from pathlib import Path

import yaml

from scripts.build_data_manifest import REQUIRED_MANIFEST_FIELDS, run as build_manifest
from scripts.clean_quantum_data import clean_records, run as clean_data
from scripts.download_quantum_data import raw_record_from_hf, stable_doc_id
from scripts.inspect_quantum_data import run as inspect_data
from scripts.sample_quantum_data import split_for_hash, split_records, run as sample_data


def _records() -> list[dict]:
    texts = [
        "Das ist ein deutscher Beispieltext mit vielen Woertern und einer klaren Struktur. "
        "Er enthaelt genug Inhalt fuer die Bereinigung und beschreibt ein technisches Projekt.",
        "Ein zweiter deutscher Text beschreibt Datenqualitaet, Filterregeln und reproduzierbare "
        "Splits. Die Quelle bleibt in den Metadaten und nicht im eigentlichen Trainingstext.",
        "Der dritte Abschnitt enthaelt Alltagssprache, Hinweise zur Schule und einfache "
        "Erklaerungen. Damit werden verschiedene deutsche Satzmuster abgedeckt.",
        "Dieser Text ist absichtlich ein weiteres Dokument fuer Validation oder Test. Er ist "
        "lang genug und enthaelt mehrere deutsche Funktionswoerter fuer die Filter.",
    ]
    out = []
    for index, text in enumerate(texts):
        digest = stable_doc_id("epfml/FineWeb2-HQ", "deu_Latn", text, f"https://example.org/{index}")
        out.append(
            {
                "id": digest,
                "source_dataset": "epfml/FineWeb2-HQ",
                "source_subset": "deu_Latn",
                "source_revision": "main",
                "source_split": "train",
                "source_index": index,
                "source_url": f"https://example.org/{index}",
                "text": text,
                "raw_text_sha256": digest,
                "raw_text_bytes": len(text.encode("utf-8")),
                "metadata": {"url": f"https://example.org/{index}"},
            }
        )
    return out


def _config(tmp_path: Path) -> Path:
    config = {
        "project": {"name": "Lumen Quantum", "dataset_name": "test-dataset", "description": "test"},
        "seed": 123,
        "source": {
            "hf_dataset": "epfml/FineWeb2-HQ",
            "hf_subset": "deu_Latn",
            "split": "train",
            "revision": "main",
            "license": "test-license",
            "citation": "test",
        },
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "cleaned_dir": str(tmp_path / "cleaned"),
            "manifest_dir": str(tmp_path / "manifests"),
            "report_dir": str(tmp_path / "reports"),
        },
        "download": {
            "streaming": True,
            "max_documents": 100000,
            "max_raw_bytes": 2147483648,
            "shuffle_buffer_size": 100,
            "text_fields": ["text"],
            "url_fields": ["url"],
        },
        "cleaning": {
            "min_chars": 80,
            "max_chars": 1000,
            "min_words": 12,
            "max_repeated_char_run": 8,
            "max_non_letter_ratio": 0.40,
            "min_german_stopword_hits": 2,
            "normalize_whitespace": True,
            "remove_exact_duplicates": True,
            "boilerplate_patterns": ["cookie", "javascript"],
        },
        "sampling": {
            "split_seed": 123,
            "train_ratio": 0.98,
            "validation_ratio": 0.01,
            "test_ratio": 0.01,
        },
        "manifest": {"version": "test-v0", "notes": "test"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_no_split_overlap_with_stable_hash_logic():
    records, _stats = clean_records(_records(), _config_dict_cleaning())
    splits = split_records(
        records,
        {"split_seed": 7, "train_ratio": 0.5, "validation_ratio": 0.25, "test_ratio": 0.25},
    )
    hash_sets = [{record["sha256"] for record in split} for split in splits.values()]
    assert hash_sets[0].isdisjoint(hash_sets[1])
    assert hash_sets[0].isdisjoint(hash_sets[2])
    assert hash_sets[1].isdisjoint(hash_sets[2])


def _config_dict_cleaning() -> dict:
    return {
        "min_chars": 80,
        "max_chars": 1000,
        "min_words": 12,
        "max_repeated_char_run": 8,
        "max_non_letter_ratio": 0.40,
        "min_german_stopword_hits": 2,
        "normalize_whitespace": True,
        "remove_exact_duplicates": True,
        "boilerplate_patterns": ["cookie", "javascript"],
    }


def test_reproducible_hash_sampling():
    digest = "a" * 64
    first = split_for_hash(digest, seed=42, train_ratio=0.98, validation_ratio=0.01)
    second = split_for_hash(digest, seed=42, train_ratio=0.98, validation_ratio=0.01)
    changed_seed = split_for_hash(digest, seed=43, train_ratio=0.98, validation_ratio=0.01)

    assert first == second
    assert changed_seed in {"train", "validation", "test"}


def test_deduplication_and_no_empty_documents():
    records = _records()
    records.append(dict(records[0]))
    records.append({**records[1], "id": "empty", "text": ""})

    cleaned, stats = clean_records(records, _config_dict_cleaning())

    assert stats["rejection_counts"]["exact_duplicate"] == 1
    assert stats["rejection_counts"]["empty_text"] == 1
    assert all(record["text"].strip() for record in cleaned)


def test_raw_record_keeps_url_outside_training_text():
    hf_record = {"text": "Das ist ein deutscher Text mit Inhalt.", "url": "https://example.org"}
    config = {
        "source": {"hf_dataset": "epfml/FineWeb2-HQ", "hf_subset": "deu_Latn", "revision": "main", "split": "train"},
        "download": {"text_fields": ["text"], "url_fields": ["url"]},
    }
    record = raw_record_from_hf(hf_record, config, index=0)

    assert record["text"] == hf_record["text"]
    assert record["source_url"] == "https://example.org"
    assert "https://example.org" not in record["text"]


def test_manifest_contains_required_fields(tmp_path):
    config_path = _config(tmp_path)
    raw_file = tmp_path / "raw" / "fineweb2_hq_deu_latn_raw.jsonl"
    _write_jsonl(raw_file, _records())
    clean_data(config_path)
    sample_data(config_path)
    inspect_data(config_path)
    manifest_json, _manifest_md = build_manifest(config_path)

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    for field in REQUIRED_MANIFEST_FIELDS:
        assert field in manifest
    assert manifest["source"]["hf_dataset"] == "epfml/FineWeb2-HQ"
    assert manifest["document_counts"]["cleaned"] > 0
