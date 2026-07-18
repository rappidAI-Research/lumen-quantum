#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Any

import numpy as np


class Uint16ShardWriter:
    def __init__(
        self,
        split: str,
        output_directory: Path,
        tokens_per_shard: int,
        target_tokens: int,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.split = split
        self.output_directory = output_directory
        self.tokens_per_shard = int(tokens_per_shard)
        self.target_tokens = int(target_tokens)

        self.output_directory.mkdir(parents=True, exist_ok=True)

        state = state or {}

        self.shard_index = int(state.get("shard_index", 0))
        self.shard_tokens = int(state.get("shard_tokens", 0))
        self.total_tokens = int(state.get("total_tokens", 0))
        self.completed_shards = list(
            state.get("completed_shards", [])
        )

        self.handle = None
        self.current_path: Path | None = None

        if self.total_tokens < self.target_tokens:
            self._reconcile_from_checkpoint()
            self._open_current_shard(
                truncate_to=state.get("current_file_size_bytes")
            )

    def _part_path(self, index: int) -> Path:
        return self.output_directory / (
            f"{self.split}-{index:05d}.bin.part"
        )

    def _final_path(self, index: int) -> Path:
        return self.output_directory / (
            f"{self.split}-{index:05d}.bin"
        )

    @staticmethod
    def _index_from_path(path: Path) -> int:
        name = path.name
        number = name.split("-")[-1].split(".")[0]
        return int(number)

    def _reconcile_from_checkpoint(self) -> None:
        current_part = self._part_path(self.shard_index)
        current_final = self._final_path(self.shard_index)

        # Ein nach dem letzten Checkpoint finalisierter Shard wird wieder
        # zur Part-Datei gemacht und anschließend auf Checkpoint-Größe
        # gekürzt.
        if current_final.exists():
            if current_part.exists():
                current_final.unlink()
            elif self.shard_tokens > 0:
                os.replace(current_final, current_part)
            else:
                current_final.unlink()

        # Alle später entstandenen Dateien gehören zu nicht
        # checkpointeten Fortschritten und müssen verworfen werden.
        for path in self.output_directory.glob(
            f"{self.split}-*.bin"
        ):
            if self._index_from_path(path) > self.shard_index:
                path.unlink()

        for path in self.output_directory.glob(
            f"{self.split}-*.bin.part"
        ):
            if self._index_from_path(path) > self.shard_index:
                path.unlink()

    def _open_current_shard(
        self,
        truncate_to: int | None = None,
    ) -> None:
        self.current_path = self._part_path(self.shard_index)
        self.current_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.handle = self.current_path.open("a+b")

        if truncate_to is not None:
            self.handle.truncate(int(truncate_to))

        self.handle.seek(0, os.SEEK_END)

        expected_size = self.shard_tokens * 2
        actual_size = self.handle.tell()

        if actual_size != expected_size:
            raise RuntimeError(
                f"{self.split}: Part-Datei hat {actual_size} Bytes, "
                f"erwartet wurden {expected_size} Bytes."
            )

    @property
    def finished(self) -> bool:
        return self.total_tokens >= self.target_tokens

    @property
    def remaining_tokens(self) -> int:
        return max(
            self.target_tokens - self.total_tokens,
            0,
        )

    def write(self, token_ids: list[int]) -> int:
        if self.finished or not token_ids:
            return 0

        remaining = min(
            len(token_ids),
            self.remaining_tokens,
        )

        values = np.asarray(
            token_ids[:remaining],
            dtype=np.uint16,
        )

        offset = 0
        written = 0

        while offset < len(values):
            shard_space = (
                self.tokens_per_shard - self.shard_tokens
            )
            chunk_size = min(
                shard_space,
                len(values) - offset,
            )

            chunk = values[offset:offset + chunk_size]
            self.handle.write(chunk.tobytes(order="C"))

            self.shard_tokens += chunk_size
            self.total_tokens += chunk_size
            written += chunk_size
            offset += chunk_size

            if (
                self.shard_tokens >= self.tokens_per_shard
                or self.finished
            ):
                self._finalize_current_shard()

        return written

    def _finalize_current_shard(self) -> None:
        if self.handle is None or self.current_path is None:
            return

        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.handle = None

        final_path = self._final_path(self.shard_index)

        if final_path.exists():
            raise FileExistsError(
                f"Zieldatei existiert bereits: {final_path}"
            )

        os.replace(self.current_path, final_path)

        self.completed_shards.append({
            "path": str(final_path),
            "tokens": self.shard_tokens,
            "bytes": final_path.stat().st_size,
        })

        self.shard_index += 1
        self.shard_tokens = 0
        self.current_path = None

        if not self.finished:
            self._open_current_shard()

    def flush(self) -> None:
        if self.handle is not None:
            self.handle.flush()
            os.fsync(self.handle.fileno())

    def state_dict(self) -> dict[str, Any]:
        self.flush()

        current_file_size = 0
        if self.current_path is not None and self.current_path.exists():
            current_file_size = self.current_path.stat().st_size

        return {
            "split": self.split,
            "shard_index": self.shard_index,
            "shard_tokens": self.shard_tokens,
            "total_tokens": self.total_tokens,
            "target_tokens": self.target_tokens,
            "tokens_per_shard": self.tokens_per_shard,
            "current_file_size_bytes": current_file_size,
            "completed_shards": self.completed_shards,
            "finished": self.finished,
        }

    def close(self) -> None:
        if self.handle is not None:
            self.flush()
            self.handle.close()
            self.handle = None

    def __enter__(self) -> "Uint16ShardWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
