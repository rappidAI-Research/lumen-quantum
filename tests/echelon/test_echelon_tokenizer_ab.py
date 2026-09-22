import json
from pathlib import Path

import pytest

from scripts.echelon_tokenizer_ab import compare_reports, load_eval_records

pytestmark = pytest.mark.unit


def _report(corpus: str, eval_sha: str, *, tokens_per_word: float) -> dict:
    metrics = {
        "tokens_per_word": tokens_per_word,
        "bytes_per_token": 3.0,
        "characters_per_token": 2.5,
        "byte_fallback_rate": 0.0,
        "roundtrip_failures": 0,
    }
    return {
        "training_corpus_sha256": corpus,
        "evaluation_corpus": {"sha256": eval_sha},
        "model_sha256": "a" * 64,
        "overall": metrics,
        "domains": {"de": metrics},
    }


def test_eval_corpus_is_structured_and_nonempty() -> None:
    records = load_eval_records(Path("data/evals/echelon_tokenizer_ab_v1.jsonl"))
    assert records
    assert {"de", "en", "code", "math"}.issubset({record["domain"] for record in records})


def test_compare_requires_identical_training_corpus() -> None:
    with pytest.raises(ValueError, match="same training corpus"):
        compare_reports(
            _report("a", "eval", tokens_per_word=1.0), _report("b", "eval", tokens_per_word=0.9)
        )


def test_compare_requires_identical_evaluation_corpus() -> None:
    with pytest.raises(ValueError, match="same evaluation corpus"):
        compare_reports(
            _report("same", "a", tokens_per_word=1.0), _report("same", "b", tokens_per_word=0.9)
        )


def test_compare_reports_deltas_without_auto_freezing() -> None:
    result = compare_reports(
        _report("same", "eval", tokens_per_word=1.2),
        _report("same", "eval", tokens_per_word=1.0),
    )
    assert result["overall"]["tokens_per_word_b_minus_a"] == pytest.approx(-0.2)
    assert result["decision_status"] == "maintainer_review_required"
    json.dumps(result)
