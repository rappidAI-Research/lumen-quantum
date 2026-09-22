"""Deterministic shared-corpus assembler for the Echelon 1B tokenizer A/B gate.

The assembler is deliberately network-free. Source-specific extraction/filtering
must stage JSONL records first. Production mode refuses every source whose
registry entry has not been explicitly approved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CorpusRecord:
    source_id: str
    record_id: str
    text: str
    text_bytes: int
    text_sha256: str
    selection_key: str
    output_key: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: YAML root must be a mapping")
    return cast("dict[str, Any]", payload)


def _stable_key(seed: str, purpose: str, source_id: str, record_id: str) -> str:
    payload = "\0".join((seed, purpose, source_id, record_id)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_inputs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        source_id, separator, raw_path = value.partition("=")
        source_id = source_id.strip()
        raw_path = raw_path.strip()
        if not separator or not source_id or not raw_path:
            raise ValueError(f"invalid --input {value!r}; expected SOURCE_ID=PATH")
        if source_id in result:
            raise ValueError(f"duplicate --input source id: {source_id}")
        result[source_id] = Path(raw_path)
    return result


def _source_map(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source registry must contain a non-empty sources list")
    result: dict[str, dict[str, Any]] = {}
    for raw in sources:
        if not isinstance(raw, dict):
            raise ValueError("source registry entries must be mappings")
        item = cast("dict[str, Any]", raw)
        source_id = str(item.get("id", "")).strip()
        if not source_id or source_id in result:
            raise ValueError(f"invalid or duplicate source id: {source_id!r}")
        result[source_id] = item
    return result


def _positive_share_sources(source_map: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        source_id: item
        for source_id, item in source_map.items()
        if Decimal(str(item.get("target_share", 0))) > 0
    }


def _byte_quotas(sources: dict[str, dict[str, Any]], target_bytes: int) -> dict[str, int]:
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    shares = {source_id: Decimal(str(item["target_share"])) for source_id, item in sources.items()}
    total_share = sum(shares.values(), Decimal(0))
    if total_share != Decimal(1):
        raise ValueError(f"positive source shares sum to {total_share}, expected 1")

    exact = {source_id: Decimal(target_bytes) * share for source_id, share in shares.items()}
    quotas = {
        source_id: int(value.to_integral_value(rounding=ROUND_FLOOR))
        for source_id, value in exact.items()
    }
    remainder = target_bytes - sum(quotas.values())
    order = sorted(sources, key=lambda source_id: (-(exact[source_id] - quotas[source_id]), source_id))
    for source_id in order[:remainder]:
        quotas[source_id] += 1
    if sum(quotas.values()) != target_bytes:
        raise RuntimeError("internal quota allocation error")
    return quotas


def _load_records(
    source_id: str,
    path: Path,
    *,
    seed: str,
    id_field: str,
    text_field: str,
) -> list[CorpusRecord]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: list[CorpusRecord] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            record_id = payload.get(id_field)
            text = payload.get(text_field)
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"{path}:{line_number}: {id_field} must be a non-empty string")
            record_id = record_id.strip()
            if record_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate record id {record_id!r}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number}: {text_field} must be non-empty text")
            seen_ids.add(record_id)
            encoded = text.encode("utf-8")
            result.append(
                CorpusRecord(
                    source_id=source_id,
                    record_id=record_id,
                    text=text,
                    text_bytes=len(encoded),
                    text_sha256=hashlib.sha256(encoded).hexdigest(),
                    selection_key=_stable_key(seed, "select", source_id, record_id),
                    output_key=_stable_key(seed, "output", source_id, record_id),
                )
            )
    if not result:
        raise ValueError(f"{path}: no usable records")
    return result


def _deduplicate(records: list[CorpusRecord]) -> tuple[list[CorpusRecord], int]:
    canonical: dict[str, CorpusRecord] = {}
    duplicate_count = 0
    for record in records:
        existing = canonical.get(record.text_sha256)
        if existing is None:
            canonical[record.text_sha256] = record
            continue
        duplicate_count += 1
        if record.selection_key < existing.selection_key:
            canonical[record.text_sha256] = record
    return list(canonical.values()), duplicate_count


def _select_records(
    records: list[CorpusRecord],
    quotas: dict[str, int],
) -> tuple[list[CorpusRecord], dict[str, dict[str, int]]]:
    by_source: dict[str, list[CorpusRecord]] = {source_id: [] for source_id in quotas}
    for record in records:
        if record.source_id in by_source:
            by_source[record.source_id].append(record)

    selected: list[CorpusRecord] = []
    stats: dict[str, dict[str, int]] = {}
    for source_id, quota in quotas.items():
        candidates = sorted(by_source[source_id], key=lambda item: item.selection_key)
        chosen: list[CorpusRecord] = []
        chosen_bytes = 0
        for record in candidates:
            if chosen_bytes >= quota:
                break
            chosen.append(record)
            chosen_bytes += record.text_bytes
        if chosen_bytes < quota:
            raise ValueError(
                f"{source_id}: staged input has {chosen_bytes} deduplicated text bytes, "
                f"below required quota {quota}"
            )
        selected.extend(chosen)
        stats[source_id] = {
            "target_text_bytes": quota,
            "selected_records": len(chosen),
            "selected_text_bytes": chosen_bytes,
            "overshoot_text_bytes": chosen_bytes - quota,
            "available_records": len(candidates),
            "available_text_bytes": sum(item.text_bytes for item in candidates),
        }
    return selected, stats


def _selected_ids_sha256(records: list[CorpusRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: (item.source_id, item.record_id)):
        digest.update(record.source_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.record_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.text_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_corpus(path: Path, records: list[CorpusRecord]) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        for record in sorted(records, key=lambda item: item.output_key):
            handle.write(record.text)
            if not record.text.endswith("\n"):
                handle.write("\n")
    temp.replace(path)


def build_corpus(
    config_path: Path,
    source_inputs: dict[str, Path],
    *,
    target_bytes: int,
    output_dir: Path,
    mode: str,
    force: bool = False,
) -> dict[str, Any]:
    if mode not in {"smoke", "production"}:
        raise ValueError("mode must be smoke or production")

    config = load_yaml(config_path)
    if config.get("project", {}).get("model_line") != "quantum-1-echelon":
        raise ValueError("wrong model_line in tokenizer corpus config")

    registry_path = ROOT / str(config["sources"]["registry"])
    registry = load_yaml(registry_path)
    source_map = _source_map(registry)
    active_sources = _positive_share_sources(source_map)

    expected_ids = set(active_sources)
    supplied_ids = set(source_inputs)
    if supplied_ids != expected_ids:
        missing = sorted(expected_ids - supplied_ids)
        extra = sorted(supplied_ids - expected_ids)
        raise ValueError(f"source inputs must match registry; missing={missing}, extra={extra}")

    if mode == "production":
        blocked = sorted(
            source_id
            for source_id, item in active_sources.items()
            if item.get("production_approved") is not True
        )
        if blocked:
            raise ValueError(f"production source gate is closed for: {', '.join(blocked)}")

    selection = config["selection"]
    seed = str(selection["seed"])
    id_field = str(selection["record_id_field"])
    text_field = str(selection["text_field"])
    quotas = _byte_quotas(active_sources, target_bytes)

    all_records: list[CorpusRecord] = []
    input_stats: dict[str, dict[str, Any]] = {}
    for source_id in sorted(active_sources):
        input_path = source_inputs[source_id]
        records = _load_records(
            source_id,
            input_path,
            seed=seed,
            id_field=id_field,
            text_field=text_field,
        )
        all_records.extend(records)
        input_stats[source_id] = {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "records": len(records),
            "text_bytes": sum(item.text_bytes for item in records),
        }

    deduplicated, duplicates_skipped = _deduplicate(all_records)
    selected, source_stats = _select_records(deduplicated, quotas)

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / str(config["output"]["corpus_filename"])
    manifest_path = output_dir / str(config["output"]["manifest_filename"])
    if not force and (corpus_path.exists() or manifest_path.exists()):
        raise FileExistsError("corpus output already exists; pass force=True/--force to replace it")

    _write_corpus(corpus_path, selected)
    manifest: dict[str, Any] = {
        "schema_version": str(config["output"]["schema_version"]),
        "model_line": "quantum-1-echelon",
        "mode": mode,
        "production_eligible": mode == "production",
        "selection_basis": str(selection["basis"]),
        "seed": seed,
        "target_text_bytes": target_bytes,
        "selected_text_bytes": sum(item.text_bytes for item in selected),
        "selected_records": len(selected),
        "exact_duplicate_records_skipped": duplicates_skipped,
        "selected_record_set_sha256": _selected_ids_sha256(selected),
        "source_registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
        },
        "inputs": input_stats,
        "sources": {
            source_id: {
                "target_share": float(active_sources[source_id]["target_share"]),
                **source_stats[source_id],
            }
            for source_id in sorted(active_sources)
        },
        "corpus": {
            "path": str(corpus_path),
            "bytes": corpus_path.stat().st_size,
            "sha256": sha256_file(corpus_path),
        },
        "token_target": {
            "minimum": int(config["target"]["production_token_range_min"]),
            "maximum": int(config["target"]["production_token_range_max"]),
            "validation": str(config["target"]["token_count_validation"]),
        },
    }
    temp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temp_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_manifest.replace(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build one deterministic shared corpus for both Echelon tokenizer candidates."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/echelon/1b/tokenizer-corpus.yaml",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="SOURCE_ID=PATH",
        help="Staged JSONL input. Repeat once for every positive-share source.",
    )
    parser.add_argument("--target-bytes", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "production"), default="production")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = build_corpus(
        args.config,
        _parse_inputs(args.input),
        target_bytes=args.target_bytes,
        output_dir=args.output_dir,
        mode=args.mode,
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
