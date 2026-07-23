"""Tokenisiert die quantum-1 Pilotdaten fuer spaeteres Training.

Dieses Skript laedt ausschliesslich den lokal selbst trainierten
SentencePiece-Tokenizer aus tokenizer/quantum-1-pilot. Es werden keine
vortrainierten Tokenizer oder Modellgewichte geladen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import sentencepiece as spm
import torch
import yaml

LOGGER = logging.getLogger("lumen.tokenize_quantum_data")
SPLITS = ("train", "validation", "test")
REQUIRED_TOKENIZER_FILES = (
    "tokenizer.model",
    "tokenizer.vocab",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer_manifest.json",
)


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


def read_json(path: str | Path) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON-Datei nicht gefunden: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def resolve_input_files(config: dict) -> dict[str, Path]:
    input_config = config["input"]
    files = {
        "train": Path(input_config["train_file"]),
        "validation": Path(input_config["validation_file"]),
        "test": Path(input_config["test_file"]),
    }
    for split, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"{split}-Eingabedatei fehlt: {path}")
    return files


def output_file_for_split(output_dir: str | Path, split: str) -> Path:
    return Path(output_dir) / f"{split}.pt"


def validate_no_overwrite(output_dir: str | Path, manifest_name: str, overwrite: bool) -> None:
    output = Path(output_dir)
    existing = [output_file_for_split(output, split) for split in SPLITS]
    existing.append(output / manifest_name)
    present = [path for path in existing if path.exists()]
    if present and not overwrite:
        rendered = ", ".join(str(path) for path in present)
        raise FileExistsError(
            "Tokenisierte Pilotdaten existieren bereits. "
            f"Nutze --overwrite, wenn du sie bewusst neu erzeugen willst: {rendered}"
        )


def load_local_sentencepiece_tokenizer(config: dict) -> tuple[spm.SentencePieceProcessor, dict]:
    tokenizer_dir = Path(config["tokenizer"]["dir"])
    if not tokenizer_dir.exists():
        raise FileNotFoundError(
            f"Tokenizer-Ordner nicht gefunden: {tokenizer_dir}. "
            "Fuehre zuerst scripts/train_quantum_tokenizer.py aus."
        )
    missing = [name for name in REQUIRED_TOKENIZER_FILES if not (tokenizer_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Tokenizer-Dateien fehlen in {tokenizer_dir}: {', '.join(missing)}"
        )

    model_path = tokenizer_dir / config["tokenizer"].get("model_file", "tokenizer.model")
    manifest_path = tokenizer_dir / config["tokenizer"].get(
        "manifest_file", "tokenizer_manifest.json"
    )
    manifest = read_json(manifest_path)
    sp = spm.SentencePieceProcessor(model_file=str(model_path))

    token_ids = manifest.get("token_ids", {})
    expected = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for token, expected_id in expected.items():
        actual_id = int(sp.piece_to_id(token))
        manifest_id = int(token_ids.get(token, actual_id))
        if actual_id != expected_id or manifest_id != expected_id:
            raise ValueError(
                f"Tokenizer-Sondertoken {token!r} muss ID {expected_id} haben, "
                f"tokenizer.model={actual_id}, Manifest={manifest_id}."
            )

    vocab_size = int(sp.get_piece_size())
    manifest_vocab = int(manifest.get("actual_vocab_size", vocab_size))
    if manifest_vocab != vocab_size:
        raise ValueError(
            f"Tokenizer-Manifest vocab_size={manifest_vocab}, tokenizer.model={vocab_size}."
        )

    return sp, manifest


def iter_jsonl_records(path: str | Path) -> Iterable[tuple[int, dict]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ungueltiges JSONL in {path}:{line_number}") from exc


def document_hash(record: dict, text: str, hash_field: str | None) -> str:
    if hash_field and isinstance(record.get(hash_field), str) and record[hash_field].strip():
        return record[hash_field].strip()
    if isinstance(record.get("sha256"), str) and record["sha256"].strip():
        return record["sha256"].strip()
    if isinstance(record.get("raw_text_sha256"), str) and record["raw_text_sha256"].strip():
        return record["raw_text_sha256"].strip()
    return sha256_text(text)


def encode_documents_for_split(
    input_file: str | Path,
    sp: spm.SentencePieceProcessor,
    text_field: str,
    hash_field: str | None,
    max_tokens: int | None,
    eos_token_id: int,
    drop_empty_documents: bool,
) -> tuple[list[int], dict]:
    tokens: list[int] = []
    doc_hashes: list[str] = []
    documents_seen = 0
    documents_used = 0
    documents_empty = 0
    documents_skipped_by_limit = 0
    documents_truncated_by_limit = 0

    for line_number, record in iter_jsonl_records(input_file):
        documents_seen += 1
        text = record.get(text_field)
        if not isinstance(text, str) or not text.strip():
            if drop_empty_documents:
                documents_empty += 1
                continue
            raise ValueError(f"Leerer oder fehlender Text in {input_file}:{line_number}")

        encoded = list(sp.encode(text.strip(), out_type=int))
        encoded.append(eos_token_id)
        if max_tokens is not None:
            remaining = int(max_tokens) - len(tokens)
            if remaining <= 0:
                documents_skipped_by_limit += 1
                continue
            if len(encoded) > remaining:
                if remaining <= 0:
                    documents_skipped_by_limit += 1
                    continue
                encoded = encoded[:remaining]
                encoded[-1] = eos_token_id
                documents_truncated_by_limit += 1

        tokens.extend(encoded)
        doc_hashes.append(document_hash(record, text, hash_field))
        documents_used += 1

        if max_tokens is not None and len(tokens) >= int(max_tokens):
            break

    stats = {
        "input_file": str(input_file),
        "input_sha256": sha256_file(input_file),
        "documents_seen": documents_seen,
        "documents_used": documents_used,
        "documents_empty": documents_empty,
        "documents_skipped_by_limit": documents_skipped_by_limit,
        "documents_truncated_by_limit": documents_truncated_by_limit,
        "document_hash_count": len(doc_hashes),
        "document_hashes_sha256": sha256_strings(sorted(doc_hashes)),
        "document_hashes": doc_hashes,
        "tokens_before_padding": len(tokens),
        "limit_tokens": max_tokens,
    }
    return tokens, stats


def pack_fixed_sequences(
    tokens: list[int],
    context_length: int,
    pad_token_id: int,
    pad_remainder: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    if context_length <= 0:
        raise ValueError("context_length muss groesser als 0 sein.")
    sequences: list[list[int]] = []
    masks: list[list[int]] = []
    remainder_tokens = len(tokens) % context_length
    padding_tokens = 0

    for start in range(0, len(tokens), context_length):
        chunk = tokens[start : start + context_length]
        if len(chunk) < context_length:
            if not pad_remainder:
                break
            padding_tokens = context_length - len(chunk)
            mask = [1] * len(chunk) + [0] * padding_tokens
            chunk = chunk + [pad_token_id] * padding_tokens
        else:
            mask = [1] * context_length
        sequences.append(chunk)
        masks.append(mask)

    if not sequences:
        input_ids = torch.empty((0, context_length), dtype=torch.long)
        attention_mask = torch.empty((0, context_length), dtype=torch.long)
    else:
        input_ids = torch.tensor(sequences, dtype=torch.long)
        attention_mask = torch.tensor(masks, dtype=torch.long)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    stats = {
        "context_length": context_length,
        "sequence_count": int(input_ids.shape[0]),
        "tokens_after_padding": int(input_ids.numel()),
        "remainder_tokens": int(remainder_tokens),
        "padding_tokens": int(padding_tokens),
        "pad_remainder": bool(pad_remainder),
    }
    return input_ids, attention_mask, labels, stats


def save_split_tensor_file(
    output_path: str | Path,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    metadata: dict,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "metadata": metadata,
        },
        path,
    )


def assert_token_ids_in_range(input_ids: torch.Tensor, vocab_size: int, split: str) -> None:
    if input_ids.numel() == 0:
        return
    minimum = int(input_ids.min().item())
    maximum = int(input_ids.max().item())
    if minimum < 0 or maximum >= vocab_size:
        raise ValueError(
            f"Token-ID ausserhalb des Vokabulars in {split}: min={minimum}, max={maximum}, vocab_size={vocab_size}"
        )


def assert_no_split_overlap(split_stats: dict[str, dict]) -> None:
    seen: dict[str, str] = {}
    for split, stats in split_stats.items():
        for digest in stats.get("document_hashes", []):
            if digest in seen:
                raise ValueError(
                    f"Split-Ueberschneidung: Dokumenthash {digest} in {seen[digest]} und {split}."
                )
            seen[digest] = split


def split_limit(config: dict, split: str, overrides: dict[str, int | None]) -> int | None:
    override = overrides.get(split)
    if override is not None:
        return int(override)
    key = f"{split}_max_tokens"
    value = config.get("limits", {}).get(key)
    return None if value is None else int(value)


def tokenize_quantum_data(
    config_path: str | Path,
    overwrite: bool = False,
    max_tokens: int | None = None,
    validation_max_tokens: int | None = None,
    test_max_tokens: int | None = None,
) -> Path:
    config = load_config(config_path)
    output_dir = Path(config["output"]["dir"])
    manifest_name = str(config["output"].get("manifest_file", "tokenization_manifest.json"))
    should_overwrite = bool(overwrite or config["output"].get("overwrite", False))
    validate_no_overwrite(output_dir, manifest_name, should_overwrite)

    sp, tokenizer_manifest = load_local_sentencepiece_tokenizer(config)
    input_files = resolve_input_files(config)
    context_length = int(config["packing"]["context_length"])
    eos_token_id = int(sp.piece_to_id("</s>"))
    pad_token_id = int(sp.piece_to_id("<pad>"))
    vocab_size = int(sp.get_piece_size())
    text_field = str(config["input"].get("text_field", "text"))
    hash_field = config["input"].get("document_hash_field")
    drop_empty_documents = bool(config["packing"].get("drop_empty_documents", True))
    pad_remainder = bool(config["packing"].get("pad_remainder", True))
    limit_overrides = {
        "train": max_tokens,
        "validation": validation_max_tokens,
        "test": test_max_tokens,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    split_stats: dict[str, dict] = {}
    output_files: dict[str, str] = {}

    for split in SPLITS:
        limit = split_limit(config, split, limit_overrides)
        tokens, stats = encode_documents_for_split(
            input_file=input_files[split],
            sp=sp,
            text_field=text_field,
            hash_field=hash_field,
            max_tokens=limit,
            eos_token_id=eos_token_id,
            drop_empty_documents=drop_empty_documents,
        )
        input_ids, attention_mask, labels, packing_stats = pack_fixed_sequences(
            tokens=tokens,
            context_length=context_length,
            pad_token_id=pad_token_id,
            pad_remainder=pad_remainder,
        )
        assert_token_ids_in_range(input_ids, vocab_size, split)
        split_metadata = {
            "split": split,
            **stats,
            **packing_stats,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "vocab_size": vocab_size,
        }
        output_path = output_file_for_split(output_dir, split)
        save_split_tensor_file(output_path, input_ids, attention_mask, labels, split_metadata)
        split_metadata["output_file"] = str(output_path)
        split_metadata["output_sha256"] = sha256_file(output_path)
        output_files[split] = str(output_path)
        split_stats[split] = split_metadata
        LOGGER.info(
            "%s tokenisiert: docs=%d tokens=%d sequences=%d padding=%d",
            split,
            split_metadata["documents_used"],
            split_metadata["tokens_before_padding"],
            split_metadata["sequence_count"],
            split_metadata["padding_tokens"],
        )

    assert_no_split_overlap(split_stats)
    for stats in split_stats.values():
        stats.pop("document_hashes", None)

    tokenizer_dir = Path(config["tokenizer"]["dir"])
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_file": str(config_path),
        "seed": int(config["seed"]),
        "dataset_name": config["project"]["dataset_name"],
        "tokenizer": {
            "dir": str(tokenizer_dir),
            "model_file": str(
                tokenizer_dir / config["tokenizer"].get("model_file", "tokenizer.model")
            ),
            "tokenizer_model_sha256": sha256_file(
                tokenizer_dir / config["tokenizer"].get("model_file", "tokenizer.model")
            ),
            "tokenizer_manifest_sha256": sha256_file(
                tokenizer_dir / config["tokenizer"].get("manifest_file", "tokenizer_manifest.json")
            ),
            "vocab_size": vocab_size,
            "token_ids": {
                "<unk>": int(sp.piece_to_id("<unk>")),
                "<s>": int(sp.piece_to_id("<s>")),
                "</s>": eos_token_id,
                "<pad>": pad_token_id,
            },
            "manifest": tokenizer_manifest,
        },
        "input_hashes": {split: sha256_file(path) for split, path in input_files.items()},
        "output_files": output_files,
        "context_length": context_length,
        "limits": {
            "train_max_tokens": split_limit(config, "train", limit_overrides),
            "validation_max_tokens": split_limit(config, "validation", limit_overrides),
            "test_max_tokens": split_limit(config, "test", limit_overrides),
        },
        "packing": {
            "add_eos_after_each_document": True,
            "pad_remainder": pad_remainder,
            "resttoken_policy": "Letzte unvollstaendige Sequenz wird mit pad_token_id aufgefuellt; Labels auf Padding sind -100.",
            "split_isolation": "Train, Validation und Test werden getrennt gelesen, gepackt und gespeichert.",
        },
        "splits": split_stats,
    }
    manifest_path = output_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Tokenisierungsmanifest geschrieben: %s", manifest_path)
    return manifest_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenisiert quantum-1 Pilotdaten mit lokalem SentencePiece-Tokenizer."
    )
    parser.add_argument("--config", default="configs/quantum_1_pilot_data.yaml")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Bestehende tokenisierte Pilotdaten ueberschreiben.",
    )
    parser.add_argument(
        "--max-tokens", type=int, help="Trainingslimit in echten Tokens vor Padding."
    )
    parser.add_argument(
        "--validation-max-tokens", type=int, help="Validation-Limit in echten Tokens vor Padding."
    )
    parser.add_argument(
        "--test-max-tokens", type=int, help="Test-Limit in echten Tokens vor Padding."
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    tokenize_quantum_data(
        config_path=args.config,
        overwrite=args.overwrite,
        max_tokens=args.max_tokens,
        validation_max_tokens=args.validation_max_tokens,
        test_max_tokens=args.test_max_tokens,
    )


if __name__ == "__main__":
    main()
