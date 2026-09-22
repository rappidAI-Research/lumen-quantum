import json
from pathlib import Path

import pytest

from scripts.echelon_tokenizer_corpus import build_corpus

pytestmark = pytest.mark.unit

SOURCE_IDS = (
    "de-fineweb2-hq",
    "en-fineweb-edu",
    "reference-knowledge",
    "math-finemath",
    "code-stack-edu",
)


def _write_inputs(root: Path, *, reverse: bool = False) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for source_id in SOURCE_IDS:
        path = root / f"{source_id}.jsonl"
        records = [
            {
                "id": f"{source_id}-{index}",
                "text": f"{source_id} record {index}: " + (chr(97 + index) * 520),
            }
            for index in range(4)
        ]
        if reverse:
            records.reverse()
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        result[source_id] = path
    return result


def test_shared_corpus_is_independent_of_input_record_order(tmp_path: Path) -> None:
    config = Path("configs/echelon/1b/tokenizer-corpus.yaml")
    first_inputs = _write_inputs(tmp_path / "first")
    second_inputs = _write_inputs(tmp_path / "second", reverse=True)

    first = build_corpus(
        config,
        first_inputs,
        target_bytes=1_000,
        output_dir=tmp_path / "out-a",
        mode="smoke",
    )
    second = build_corpus(
        config,
        second_inputs,
        target_bytes=1_000,
        output_dir=tmp_path / "out-b",
        mode="smoke",
    )

    assert first["corpus"]["sha256"] == second["corpus"]["sha256"]
    assert first["selected_record_set_sha256"] == second["selected_record_set_sha256"]
    assert first["production_eligible"] is False
    assert sum(item["target_text_bytes"] for item in first["sources"].values()) == 1_000
    assert all(
        item["selected_text_bytes"] >= item["target_text_bytes"]
        for item in first["sources"].values()
    )


def test_production_mode_refuses_unapproved_source_registry(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path / "inputs")

    with pytest.raises(ValueError, match="production source gate is closed"):
        build_corpus(
            Path("configs/echelon/1b/tokenizer-corpus.yaml"),
            inputs,
            target_bytes=1_000,
            output_dir=tmp_path / "production",
            mode="production",
        )


def test_shared_corpus_requires_every_positive_share_source(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path / "inputs")
    inputs.pop("reference-knowledge")

    with pytest.raises(ValueError, match="source inputs must match registry"):
        build_corpus(
            Path("configs/echelon/1b/tokenizer-corpus.yaml"),
            inputs,
            target_bytes=1_000,
            output_dir=tmp_path / "missing",
            mode="smoke",
        )
