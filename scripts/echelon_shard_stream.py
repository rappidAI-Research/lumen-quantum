"""Deterministic memory-mapped token stream for Quantum 1 Echelon.

The stream reads canonical uint16 token shards without loading the corpus into
RAM. Resume state is a global token offset, so reconnects or process restarts do
not depend on Python DataLoader internals.

This module does not download data, start training or access AWS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np


SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class ShardRecord:
    path: Path
    tokens: int
    bytes: int
    sha256: str


@dataclass(frozen=True)
class StreamCursor:
    global_token_offset: int
    sequences_emitted: int

    def as_dict(self) -> dict[str, int]:
        return {
            "global_token_offset": self.global_token_offset,
            "sequences_emitted": self.sequences_emitted,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path, *, verify_files: bool = False) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("shard manifest root must be a JSON object")
    manifest = cast("dict[str, Any]", payload)

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported shard manifest schema_version")
    if manifest.get("model_line") != "quantum-1-echelon":
        raise ValueError("manifest model_line must be quantum-1-echelon")
    if manifest.get("token_dtype") != "uint16":
        raise ValueError("only uint16 shard manifests are supported")

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("manifest must contain at least one shard")

    root = path.parent
    total = 0
    for index, item in enumerate(shards):
        if not isinstance(item, dict):
            raise ValueError(f"shards[{index}] must be an object")
        shard_path = Path(str(item["path"]))
        if not shard_path.is_absolute():
            shard_path = (root / shard_path).resolve()
        tokens = int(item["tokens"])
        byte_count = int(item["bytes"])
        checksum = str(item["sha256"])

        if tokens <= 0:
            raise ValueError(f"shards[{index}].tokens must be positive")
        if byte_count != tokens * 2:
            raise ValueError(
                f"shards[{index}] byte count {byte_count} does not match uint16 tokens {tokens}"
            )
        if len(checksum) != 64 or any(ch not in "0123456789abcdef" for ch in checksum):
            raise ValueError(f"shards[{index}].sha256 must be lowercase SHA-256")

        if verify_files:
            if not shard_path.is_file():
                raise FileNotFoundError(shard_path)
            if shard_path.stat().st_size != byte_count:
                raise ValueError(f"size mismatch for {shard_path}")
            if sha256_file(shard_path) != checksum:
                raise ValueError(f"checksum mismatch for {shard_path}")

        item["_resolved_path"] = str(shard_path)
        total += tokens

    if int(manifest.get("total_tokens", -1)) != total:
        raise ValueError(
            f"manifest total_tokens={manifest.get('total_tokens')} but shard sum is {total}"
        )
    return manifest


class ShardedTokenStream:
    """Sequential virtual token array spanning many uint16 shard files."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        context_length: int | None = None,
        start_token_offset: int = 0,
        verify_files: bool = False,
    ) -> None:
        self.manifest_path = manifest_path
        manifest = load_manifest(manifest_path, verify_files=verify_files)
        self.context_length = int(context_length or manifest["context_length"])
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")

        self.records: list[ShardRecord] = []
        self._starts: list[int] = []
        running = 0
        for item in manifest["shards"]:
            self._starts.append(running)
            self.records.append(
                ShardRecord(
                    path=Path(item["_resolved_path"]),
                    tokens=int(item["tokens"]),
                    bytes=int(item["bytes"]),
                    sha256=str(item["sha256"]),
                )
            )
            running += int(item["tokens"])
        self.total_tokens = running

        self._offset = int(start_token_offset)
        if self._offset < 0 or self._offset > self.total_tokens:
            raise ValueError("start_token_offset outside manifest token range")
        if self._offset % self.context_length != 0:
            raise ValueError(
                "start_token_offset must align to context_length for exact sequence resume"
            )

        self._sequences_emitted = self._offset // self.context_length
        self._cached_index: int | None = None
        self._cached_mmap: np.memmap | None = None

    @property
    def cursor(self) -> StreamCursor:
        return StreamCursor(
            global_token_offset=self._offset,
            sequences_emitted=self._sequences_emitted,
        )

    @property
    def remaining_full_sequences(self) -> int:
        return (self.total_tokens - self._offset) // self.context_length

    def _locate(self, global_offset: int) -> tuple[int, int]:
        if global_offset < 0 or global_offset >= self.total_tokens:
            raise IndexError(global_offset)
        for index in range(len(self.records) - 1, -1, -1):
            start = self._starts[index]
            if global_offset >= start:
                return index, global_offset - start
        raise RuntimeError("failed to locate token offset")

    def _mmap(self, index: int) -> np.memmap:
        if self._cached_index == index and self._cached_mmap is not None:
            return self._cached_mmap
        record = self.records[index]
        array = np.memmap(record.path, dtype=np.uint16, mode="r", shape=(record.tokens,))
        self._cached_index = index
        self._cached_mmap = array
        return array

    def read_tokens(self, count: int) -> np.ndarray:
        if count <= 0:
            raise ValueError("count must be positive")
        if self._offset + count > self.total_tokens:
            raise EOFError("not enough tokens remain")

        output = np.empty(count, dtype=np.uint16)
        written = 0
        position = self._offset

        while written < count:
            shard_index, shard_offset = self._locate(position)
            record = self.records[shard_index]
            available = record.tokens - shard_offset
            take = min(count - written, available)
            mmap = self._mmap(shard_index)
            output[written : written + take] = mmap[shard_offset : shard_offset + take]
            written += take
            position += take

        self._offset = position
        return output

    def next_sequence(self) -> np.ndarray:
        sequence = self.read_tokens(self.context_length)
        self._sequences_emitted += 1
        return sequence

    def __iter__(self) -> Iterator[np.ndarray]:
        while self.remaining_full_sequences > 0:
            yield self.next_sequence()


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect an Echelon token-shard stream.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--start-token-offset", type=int, default=0)
    parser.add_argument("--take-sequences", type=int, default=0)
    arguments = parser.parse_args()

    stream = ShardedTokenStream(
        arguments.manifest,
        start_token_offset=arguments.start_token_offset,
        verify_files=arguments.verify_files,
    )
    result: dict[str, Any] = {
        "total_tokens": stream.total_tokens,
        "context_length": stream.context_length,
        "remaining_full_sequences": stream.remaining_full_sequences,
        "cursor_before": stream.cursor.as_dict(),
    }
    for _ in range(arguments.take_sequences):
        if stream.remaining_full_sequences <= 0:
            break
        stream.next_sequence()
    result["cursor_after"] = stream.cursor.as_dict()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
