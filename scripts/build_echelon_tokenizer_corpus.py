#!/usr/bin/env python3

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset


OUTPUT = Path("data/echelon/tokenizer_training/training_text.txt")
MANIFEST = Path("data/echelon/tokenizer_training/manifest.json")
TARGET_BYTES = 512 * 1024 * 1024
SEED = 20260718

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MANY_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS.sub("", text)
    text = MANY_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "epfml/FineWeb2-HQ",
        "deu_Latn",
        split="train",
        streaming=True,
    ).shuffle(seed=SEED, buffer_size=10_000)

    documents_seen = 0
    documents_written = 0
    bytes_written = 0

    with OUTPUT.open("w", encoding="utf-8") as output:
        for sample in dataset:
            documents_seen += 1
            text = clean_text(sample.get("text", ""))

            if len(text) < 200:
                continue

            encoded_size = len(text.encode("utf-8")) + 2

            if bytes_written + encoded_size > TARGET_BYTES:
                break

            output.write(text)
            output.write("\n\n")

            documents_written += 1
            bytes_written += encoded_size

            if documents_written % 10_000 == 0:
                print(
                    f"Dokumente: {documents_written:,} | "
                    f"Größe: {bytes_written / 1024**2:.1f} MiB",
                    flush=True,
                )

    sha256 = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()

    manifest = {
        "dataset": "epfml/FineWeb2-HQ",
        "configuration": "deu_Latn",
        "streaming": True,
        "shuffle_seed": SEED,
        "shuffle_buffer": 10_000,
        "target_bytes": TARGET_BYTES,
        "written_bytes": bytes_written,
        "documents_seen": documents_seen,
        "documents_written": documents_written,
        "sha256": sha256,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Korpus fertig.")
    print(f"Dokumente: {documents_written:,}")
    print(f"Größe: {bytes_written / 1024**2:.1f} MiB")
    print(f"SHA-256: {sha256}")


if __name__ == "__main__":
    main()
