#!/usr/bin/env python3

import gc
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sentencepiece as spm
import yaml
from datasets import load_dataset

from echelon_quality_filters import (
    alpha_ratio,
    boilerplate_match_count,
    duplicate_line_ratio,
    metadata_rejection,
    repeated_ngram_ratio,
    words,
)
from echelon_shard_writer import Uint16ShardWriter


CONFIG_PATH = Path(
    os.environ.get(
        "ECHELON_GARDEN_CONFIG",
        "configs/echelon/garden_production.yaml",
    )
)

CONTROL_CHARS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)
EXCESSIVE_BLANK_LINES = re.compile(r"\n{3,}")
URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def config_sha256(config: dict[str, Any]) -> str:
    serialized = yaml.safe_dump(
        config,
        allow_unicode=True,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(serialized).hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS.sub("", text)
    text = EXCESSIVE_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def ratio(count: int, total: int) -> float:
    return count / max(total, 1)


def stable_split(fingerprint: str, config: dict) -> str:
    splits = config["splits"]
    bucket = (
        int(fingerprint[:16], 16)
        % int(splits["total_buckets"])
    )

    train_limit = int(splits["train_buckets"])
    validation_limit = (
        train_limit + int(splits["validation_buckets"])
    )

    if bucket < train_limit:
        return "train"

    if bucket < validation_limit:
        return "validation"

    return "test"


def normalized_fingerprint(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def quality_rejection(text: str, config: dict) -> str | None:
    filters = config["filters"]
    length = len(text)

    if length < int(filters["minimum_characters"]):
        return "too_short"

    if length > int(filters["maximum_characters"]):
        return "too_long"

    url_characters = sum(
        len(match.group(0))
        for match in URL_PATTERN.finditer(text)
    )
    if ratio(url_characters, length) > float(
        filters["maximum_url_ratio"]
    ):
        return "url_ratio"

    digit_count = sum(character.isdigit() for character in text)
    if ratio(digit_count, length) > float(
        filters["maximum_digit_ratio"]
    ):
        return "digit_ratio"

    symbol_count = sum(
        not character.isalnum() and not character.isspace()
        for character in text
    )
    if ratio(symbol_count, length) > float(
        filters["maximum_symbol_ratio"]
    ):
        return "symbol_ratio"

    if len(words(text)) < int(filters["minimum_words"]):
        return "too_few_words"

    if alpha_ratio(text) < float(filters["minimum_alpha_ratio"]):
        return "alpha_ratio"

    if duplicate_line_ratio(text) > float(
        filters["maximum_duplicate_line_ratio"]
    ):
        return "duplicate_lines"

    if repeated_ngram_ratio(text, n=5) > float(
        filters["maximum_repeated_5gram_ratio"]
    ):
        return "repeated_5grams"

    if boilerplate_match_count(text) > int(
        filters["maximum_boilerplate_matches"]
    ):
        return "boilerplate"

    return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary_path, path)


def load_checkpoint(
    path: Path,
    expected_config_hash: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None

    checkpoint = json.loads(
        path.read_text(encoding="utf-8")
    )

    stored_hash = checkpoint.get("config_sha256")

    if stored_hash != expected_config_hash:
        raise RuntimeError(
            "Checkpoint und Produktionskonfiguration stimmen "
            "nicht überein."
        )

    return checkpoint


def quality_rejection(text: str, config: dict) -> str | None:
    filters = config["filters"]
    length = len(text)

    if length < int(filters["minimum_characters"]):
        return "too_short"

    if length > int(filters["maximum_characters"]):
        return "too_long"

    url_characters = sum(
        len(match.group(0))
        for match in URL_PATTERN.finditer(text)
    )
    if ratio(url_characters, length) > float(
        filters["maximum_url_ratio"]
    ):
        return "url_ratio"

    digit_count = sum(character.isdigit() for character in text)
    if ratio(digit_count, length) > float(
        filters["maximum_digit_ratio"]
    ):
        return "digit_ratio"

    symbol_count = sum(
        not character.isalnum() and not character.isspace()
        for character in text
    )
    if ratio(symbol_count, length) > float(
        filters["maximum_symbol_ratio"]
    ):
        return "symbol_ratio"

    if len(words(text)) < int(filters["minimum_words"]):
        return "too_few_words"

    if alpha_ratio(text) < float(filters["minimum_alpha_ratio"]):
        return "alpha_ratio"

    if duplicate_line_ratio(text) > float(
        filters["maximum_duplicate_line_ratio"]
    ):
        return "duplicate_lines"

    if repeated_ngram_ratio(text, n=5) > float(
        filters["maximum_repeated_5gram_ratio"]
    ):
        return "repeated_5grams"

    if boilerplate_match_count(text) > int(
        filters["maximum_boilerplate_matches"]
    ):
        return "boilerplate"

    return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary_path, path)


def load_checkpoint(
    path: Path,
    expected_config_hash: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None

    checkpoint = json.loads(
        path.read_text(encoding="utf-8")
    )

    stored_hash = checkpoint.get("config_sha256")

    if stored_hash != expected_config_hash:
        raise RuntimeError(
            "Checkpoint und Produktionskonfiguration stimmen "
            "nicht überein."
        )

    return checkpoint


SPLITS = ("train", "validation", "test")


def target_tokens_for_split(
    split: str,
    config: dict[str, Any],
) -> int:
    key = {
        "train": "train_tokens",
        "validation": "validation_tokens",
        "test": "test_tokens",
    }[split]

    return int(config["targets"][key])


def create_writers(
    config: dict[str, Any],
    checkpoint: dict[str, Any] | None,
) -> dict[str, Uint16ShardWriter]:
    output_root = Path(config["output"]["root"])
    tokenized_root = output_root / "tokenized"
    writer_states = (
        checkpoint.get("writers", {})
        if checkpoint is not None
        else {}
    )

    writers = {}

    for split in SPLITS:
        writers[split] = Uint16ShardWriter(
            split=split,
            output_directory=tokenized_root / split,
            tokens_per_shard=int(
                config["targets"]["tokens_per_shard"]
            ),
            target_tokens=target_tokens_for_split(
                split,
                config,
            ),
            state=writer_states.get(split),
        )

    return writers


def load_source_dataset(
    config: dict[str, Any],
    checkpoint: dict[str, Any] | None,
):
    source = config["source"]

    dataset = load_dataset(
        source["dataset"],
        source["configuration"],
        split=source["split"],
        streaming=bool(source["streaming"]),
        revision=source["revision"],
    )

    resume_mode = "fresh"
    consumed = 0

    if checkpoint is not None:
        consumed = int(
            checkpoint.get("source_documents_consumed", 0)
        )
        source_state = checkpoint.get("source_state")

        if (
            source_state is not None
            and hasattr(dataset, "load_state_dict")
        ):
            dataset.load_state_dict(source_state)
            resume_mode = "state_dict"

        elif consumed > 0:
            dataset = dataset.skip(consumed)
            resume_mode = "skip"

    return dataset, resume_mode, consumed


def initialize_runtime():
    config = load_config()
    config_hash = config_sha256(config)

    output_root = Path(config["output"]["root"])
    checkpoint_path = Path(config["output"]["checkpoint"])
    manifest_path = Path(config["output"]["manifest"])

    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = None

    if bool(config["runtime"]["resume"]):
        checkpoint = load_checkpoint(
            checkpoint_path,
            config_hash,
        )

    tokenizer = spm.SentencePieceProcessor(
        model_file=config["tokenizer"]["model"]
    )

    if tokenizer.get_piece_size() > 65536:
        raise RuntimeError(
            "Tokenizer-Vokabular passt nicht in uint16."
        )

    writers = create_writers(config, checkpoint)

    print("Lade Dataset-Stream ...", flush=True)

    dataset, resume_mode, consumed = load_source_dataset(
        config,
        checkpoint,
    )

    counters = Counter(
        checkpoint.get("counters", {})
        if checkpoint is not None
        else {}
    )

    started_utc = (
        checkpoint.get("started_utc")
        if checkpoint is not None
        else utc_now()
    )

    print("=== GARDEN-PRODUKTION INITIALISIERT ===")
    print("Dataset:", config["source"]["dataset"])
    print("Revision:", config["source"]["revision"])
    print("Vokabular:", tokenizer.get_piece_size())
    print("Wiederaufnahmemodus:", resume_mode)
    print("Bereits konsumierte Dokumente:", consumed)

    for split in SPLITS:
        print(
            f"{split}: "
            f"{writers[split].total_tokens:,} / "
            f"{writers[split].target_tokens:,} Tokens"
        )

    return {
        "config": config,
        "config_hash": config_hash,
        "checkpoint_path": checkpoint_path,
        "manifest_path": manifest_path,
        "checkpoint": checkpoint,
        "tokenizer": tokenizer,
        "writers": writers,
        "dataset": dataset,
        "counters": counters,
        "started_utc": started_utc,
        "source_documents_consumed": consumed,
        "resume_mode": resume_mode,
        "source_state": (
            checkpoint.get("source_state")
            if checkpoint is not None
            else None
        ),
    }


def all_writers_finished(
    writers: dict[str, Uint16ShardWriter],
) -> bool:
    return all(
        writers[split].finished
        for split in SPLITS
    )


def build_checkpoint_payload(
    runtime: dict[str, Any],
    complete: bool = False,
) -> dict[str, Any]:
    writers = runtime["writers"]

    return {
        "checkpoint_version": 1,
        "pipeline": "quantum-1-echelon-garden-production",
        "config_sha256": runtime["config_hash"],
        "dataset": runtime["config"]["source"]["dataset"],
        "dataset_configuration": runtime["config"]["source"][
            "configuration"
        ],
        "dataset_revision": runtime["config"]["source"][
            "revision"
        ],
        "started_utc": runtime["started_utc"],
        "updated_utc": utc_now(),
        "complete": complete,
        "source_documents_consumed": runtime[
            "source_documents_consumed"
        ],
        "resume_strategy": (
            "dataset_state_dict"
            if runtime.get("source_state") is not None
            else "deterministic_skip"
        ),
        "source_state": runtime.get("source_state"),
        "counters": dict(runtime["counters"]),
        "writers": {
            split: writers[split].state_dict()
            for split in SPLITS
        },
    }


def capture_source_state(
    runtime: dict[str, Any],
) -> None:
    dataset = runtime.get("dataset")

    if dataset is None:
        return

    state_method = getattr(dataset, "state_dict", None)

    if not callable(state_method):
        return

    state = state_method()

    # Prüft gleichzeitig, ob der Zustand im JSON-Checkpoint
    # gespeichert werden kann.
    json.dumps(state)
    runtime["source_state"] = state


def save_runtime_checkpoint(
    runtime: dict[str, Any],
    complete: bool = False,
) -> None:
    capture_source_state(runtime)

    payload = build_checkpoint_payload(
        runtime,
        complete=complete,
    )

    atomic_write_json(
        runtime["checkpoint_path"],
        payload,
    )


def print_progress(
    runtime: dict[str, Any],
    elapsed_seconds: float,
) -> None:
    counters = runtime["counters"]
    consumed = runtime["source_documents_consumed"]

    documents_per_second = consumed / max(
        elapsed_seconds,
        0.001,
    )

    print("\n=== GARDEN-FORTSCHRITT ===", flush=True)
    print(
        f"Dokumente konsumiert: {consumed:,}",
        flush=True,
    )
    print(
        f"Dokumente akzeptiert: "
        f"{counters.get('documents_accepted', 0):,}",
        flush=True,
    )
    print(
        f"Geschwindigkeit: "
        f"{documents_per_second:,.2f} Dokumente/s",
        flush=True,
    )

    for split in SPLITS:
        writer = runtime["writers"][split]
        percentage = (
            100.0
            * writer.total_tokens
            / max(writer.target_tokens, 1)
        )

        print(
            f"{split}: {writer.total_tokens:,} / "
            f"{writer.target_tokens:,} Tokens "
            f"({percentage:.4f} %)",
            flush=True,
        )


def process_dataset(runtime: dict[str, Any]) -> None:
    config = runtime["config"]
    tokenizer = runtime["tokenizer"]
    writers = runtime["writers"]
    counters = runtime["counters"]

    checkpoint_interval = int(
        config["runtime"]["checkpoint_every_documents"]
    )
    progress_interval = int(
        config["runtime"]["progress_every_documents"]
    )

    start_time = time.monotonic()
    controlled_stop_after = int(
        os.environ.get(
            "ECHELON_TEST_STOP_AFTER_DOCUMENTS",
            "0",
        )
    )

    dataset_iterator = iter(runtime["dataset"])

    try:
        for sample in dataset_iterator:
            if all_writers_finished(writers):
                break

            runtime["source_documents_consumed"] += 1
            consumed = runtime["source_documents_consumed"]

            counters["documents_seen"] += 1

            metadata_reason = metadata_rejection(
                sample,
                config,
            )

            if metadata_reason is not None:
                counters[
                    f"rejected_{metadata_reason}"
                ] += 1
            else:
                text = clean_text(
                    str(sample.get("text", ""))
                )

                quality_reason = quality_rejection(
                    text,
                    config,
                )

                if quality_reason is not None:
                    counters[
                        f"rejected_{quality_reason}"
                    ] += 1
                else:
                    fingerprint = normalized_fingerprint(
                        text
                    )
                    split = stable_split(
                        fingerprint,
                        config,
                    )

                    if writers[split].finished:
                        counters[
                            f"skipped_finished_{split}"
                        ] += 1
                    else:
                        token_ids = tokenizer.encode(
                            text,
                            out_type=int,
                            add_bos=False,
                            add_eos=True,
                        )

                        written = writers[split].write(
                            token_ids
                        )

                        if written > 0:
                            counters[
                                "documents_accepted"
                            ] += 1
                            counters[
                                f"documents_accepted_{split}"
                            ] += 1
                            counters[
                                f"tokens_written_{split}"
                            ] += written

                            if written < len(token_ids):
                                counters[
                                    f"documents_truncated_{split}"
                                ] += 1

            if consumed % progress_interval == 0:
                print_progress(
                    runtime,
                    time.monotonic() - start_time,
                )

            if consumed % checkpoint_interval == 0:
                save_runtime_checkpoint(runtime)
                print(
                    f"Checkpoint gespeichert nach "
                    f"{consumed:,} Dokumenten.",
                    flush=True,
                )

            if (
                controlled_stop_after > 0
                and consumed >= controlled_stop_after
            ):
                save_runtime_checkpoint(runtime)
                runtime["controlled_test_stop"] = True
                print(
                    f"Kontrollierter Teststopp nach "
                    f"{consumed:,} Dokumenten.",
                    flush=True,
                )
                return

    except KeyboardInterrupt:
        print(
            "\nAbbruch erkannt. Speichere Checkpoint …",
            flush=True,
        )
        save_runtime_checkpoint(runtime)
        raise

    except BaseException:
        print(
            "\nFehler erkannt. Speichere Checkpoint …",
            flush=True,
        )
        save_runtime_checkpoint(runtime)
        raise

    finally:
        close_method = getattr(
            dataset_iterator,
            "close",
            None,
        )

        if callable(close_method):
            try:
                close_method()
            except Exception as error:
                print(
                    "Warnung beim Schließen des Dataset-Streams: "
                    f"{error}",
                    flush=True,
                )

        runtime["dataset"] = None
        del dataset_iterator
        gc.collect()


def build_manifest(
    runtime: dict[str, Any],
    complete: bool,
    elapsed_seconds: float,
) -> dict[str, Any]:
    writers = runtime["writers"]

    return {
        "manifest_version": 1,
        "pipeline": "quantum-1-echelon-garden-production",
        "complete": complete,
        "started_utc": runtime["started_utc"],
        "finished_utc": utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "config_path": str(CONFIG_PATH),
        "config_sha256": runtime["config_hash"],
        "dataset": {
            "name": runtime["config"]["source"]["dataset"],
            "configuration": runtime["config"]["source"][
                "configuration"
            ],
            "split": runtime["config"]["source"]["split"],
            "revision": runtime["config"]["source"][
                "revision"
            ],
        },
        "tokenizer": {
            "model": runtime["config"]["tokenizer"]["model"],
            "vocabulary_size": runtime[
                "tokenizer"
            ].get_piece_size(),
        },
        "source_documents_consumed": runtime[
            "source_documents_consumed"
        ],
        "counters": dict(runtime["counters"]),
        "splits": {
            split: writers[split].state_dict()
            for split in SPLITS
        },
    }


def close_writers(
    writers: dict[str, Uint16ShardWriter],
) -> None:
    errors = []

    for split in SPLITS:
        try:
            writers[split].close()
        except Exception as error:
            errors.append(f"{split}: {error}")

    if errors:
        raise RuntimeError(
            "Fehler beim Schließen der Shard-Writer: "
            + "; ".join(errors)
        )


def main() -> None:
    runtime = initialize_runtime()
    start_time = time.monotonic()
    completed = False

    try:
        if all_writers_finished(runtime["writers"]):
            print(
                "Alle Tokenziele waren bereits erreicht.",
                flush=True,
            )
        else:
            process_dataset(runtime)

        if runtime.get("controlled_test_stop"):
            elapsed_seconds = time.monotonic() - start_time

            manifest = build_manifest(
                runtime,
                complete=False,
                elapsed_seconds=elapsed_seconds,
            )

            atomic_write_json(
                runtime["manifest_path"],
                manifest,
            )

            print_progress(runtime, elapsed_seconds)
            print(
                "\nKontrollierter Resume-Test-Stopp abgeschlossen.",
                flush=True,
            )
            return

        completed = all_writers_finished(
            runtime["writers"]
        )

        save_runtime_checkpoint(
            runtime,
            complete=completed,
        )

        elapsed_seconds = time.monotonic() - start_time

        manifest = build_manifest(
            runtime,
            complete=completed,
            elapsed_seconds=elapsed_seconds,
        )

        atomic_write_json(
            runtime["manifest_path"],
            manifest,
        )

        print_progress(runtime, elapsed_seconds)

        if not completed:
            raise RuntimeError(
                "Der Quelldatensatz endete, bevor alle Tokenziele "
                "erreicht wurden."
            )

        print(
            "\nGarden-Produktionspipeline vollständig abgeschlossen.",
            flush=True,
        )
        print(
            f"Manifest: {runtime['manifest_path']}",
            flush=True,
        )

    finally:
        close_writers(runtime["writers"])


if __name__ == "__main__":
    import sys

    main()

    # Alle eigenen Dateien wurden in main() bereits gespeichert und
    # geschlossen. os._exit verhindert einen bekannten Absturz beim
    # Interpreter-Shutdown durch verbleibende HF-Streaming-Threads.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
