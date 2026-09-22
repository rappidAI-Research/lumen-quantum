"""Reproducible 32K/48K tokenizer A/B tooling for Quantum 1 Echelon.

Both candidates must be trained from the exact same immutable local corpus. The
tool records corpus/config checksums and produces domain-level efficiency and
round-trip evidence. It deliberately does not auto-freeze a winner.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import sentencepiece as spm
import yaml


BYTE_PIECE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: config root must be a mapping")
    if payload.get("project", {}).get("model_line") != "quantum-1-echelon":
        raise ValueError(f"{path}: wrong model_line")
    return payload


def config_sha256(path: Path) -> str:
    payload = load_config(path)
    normalized = yaml.safe_dump(payload, allow_unicode=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def corpus_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    bytes_total = path.stat().st_size
    lines = 0
    nonempty_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            lines += 1
            if line.strip():
                nonempty_lines += 1
    if nonempty_lines == 0:
        raise ValueError("tokenizer corpus contains no non-empty lines")
    return {
        "path": str(path),
        "bytes": bytes_total,
        "lines": lines,
        "nonempty_lines": nonempty_lines,
        "sha256": sha256_file(path),
    }


def _special_symbols(config: dict[str, Any]) -> list[str]:
    special = config["special_tokens"]
    return [
        str(special["system_token"]),
        str(special["user_token"]),
        str(special["assistant_token"]),
        str(special["end_token"]),
    ]


def train_candidate(config_path: Path, corpus_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    corpus = corpus_stats(corpus_path)
    tokenizer = config["tokenizer"]
    output = config["output"]
    output_dir = Path(output["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / str(output["model_prefix"])

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(prefix),
        vocab_size=int(tokenizer["vocab_size"]),
        model_type="bpe",
        character_coverage=float(tokenizer["character_coverage"]),
        byte_fallback=bool(tokenizer["byte_fallback"]),
        split_digits=bool(tokenizer["split_digits"]),
        normalization_rule_name=str(tokenizer["normalization_rule_name"]),
        allow_whitespace_only_pieces=bool(tokenizer["allow_whitespace_only_pieces"]),
        remove_extra_whitespaces=bool(tokenizer["remove_extra_whitespaces"]),
        hard_vocab_limit=False,
        pad_id=int(config["expected_ids"]["pad_token_id"]),
        bos_id=int(config["expected_ids"]["bos_token_id"]),
        eos_id=int(config["expected_ids"]["eos_token_id"]),
        unk_id=int(config["expected_ids"]["unk_token_id"]),
        user_defined_symbols=_special_symbols(config),
    )

    model_path = prefix.with_suffix(".model")
    vocab_path = prefix.with_suffix(".vocab")
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    expected = config["expected_ids"]
    special = config["special_tokens"]
    id_checks = {
        "pad_token_id": processor.piece_to_id(str(special["pad_token"])),
        "bos_token_id": processor.piece_to_id(str(special["bos_token"])),
        "eos_token_id": processor.piece_to_id(str(special["eos_token"])),
        "unk_token_id": processor.piece_to_id(str(special["unk_token"])),
    }
    for key, actual in id_checks.items():
        if int(actual) != int(expected[key]):
            raise RuntimeError(f"{key}: trained={actual}, expected={expected[key]}")

    for symbol in _special_symbols(config):
        token_id = int(processor.piece_to_id(symbol))
        if token_id < 0 or token_id == int(processor.unk_id()):
            raise RuntimeError(f"missing user-defined symbol: {symbol}")

    manifest = {
        "schema_version": "1.0.0",
        "model_line": "quantum-1-echelon",
        "candidate": str(config["project"]["candidate"]),
        "requested_vocab_size": int(tokenizer["vocab_size"]),
        "actual_vocab_size": int(processor.get_piece_size()),
        "config_path": str(config_path),
        "config_sha256": config_sha256(config_path),
        "training_corpus": corpus,
        "sentencepiece_version": getattr(spm, "__version__", "unknown"),
        "model": {
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
        },
        "vocab": {
            "path": str(vocab_path),
            "bytes": vocab_path.stat().st_size,
            "sha256": sha256_file(vocab_path),
        },
        "special_token_ids": id_checks,
    }
    manifest_path = output_dir / "tokenizer-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_eval_records(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: record must be an object")
        domain = payload.get("domain")
        text = payload.get("text")
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError(f"{path}:{line_number}: domain must be non-empty")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{path}:{line_number}: text must be non-empty")
        records.append({"domain": domain.strip(), "text": text})
    if not records:
        raise ValueError("evaluation corpus contains no records")
    return records


def _record_metrics(processor: Any, text: str) -> dict[str, int]:
    ids = processor.encode(text, out_type=int)
    pieces = processor.encode(text, out_type=str)
    decoded = processor.decode(ids)
    words = len(text.split())
    return {
        "characters": len(text),
        "bytes": len(text.encode("utf-8")),
        "words": words,
        "tokens": len(ids),
        "byte_fallback_tokens": sum(bool(BYTE_PIECE.fullmatch(piece)) for piece in pieces),
        "roundtrip_failure": int(decoded != text),
    }


def _summarize(items: list[dict[str, int]]) -> dict[str, Any]:
    totals = {
        key: sum(item[key] for item in items)
        for key in (
            "characters",
            "bytes",
            "words",
            "tokens",
            "byte_fallback_tokens",
            "roundtrip_failure",
        )
    }
    token_count = max(totals["tokens"], 1)
    word_count = max(totals["words"], 1)
    return {
        **totals,
        "bytes_per_token": totals["bytes"] / token_count,
        "characters_per_token": totals["characters"] / token_count,
        "tokens_per_word": totals["tokens"] / word_count,
        "byte_fallback_rate": totals["byte_fallback_tokens"] / token_count,
        "roundtrip_failures": totals["roundtrip_failure"],
    }


def evaluate_candidate(
    model_path: Path,
    eval_path: Path,
    *,
    tokenizer_manifest: Path | None = None,
) -> dict[str, Any]:
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    records = load_eval_records(eval_path)
    grouped: dict[str, list[dict[str, int]]] = defaultdict(list)
    all_items: list[dict[str, int]] = []
    for record in records:
        metrics = _record_metrics(processor, record["text"])
        grouped[record["domain"]].append(metrics)
        all_items.append(metrics)

    training_corpus_sha256 = None
    manifest_sha256 = None
    if tokenizer_manifest is not None:
        payload = json.loads(tokenizer_manifest.read_text(encoding="utf-8"))
        training_corpus_sha256 = payload.get("training_corpus", {}).get("sha256")
        manifest_sha256 = sha256_file(tokenizer_manifest)

    return {
        "schema_version": "1.0.0",
        "model_line": "quantum-1-echelon",
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "tokenizer_manifest_sha256": manifest_sha256,
        "training_corpus_sha256": training_corpus_sha256,
        "evaluation_corpus": {
            "path": str(eval_path),
            "sha256": sha256_file(eval_path),
            "records": len(records),
        },
        "overall": _summarize(all_items),
        "domains": {domain: _summarize(items) for domain, items in sorted(grouped.items())},
    }


def compare_reports(report_a: dict[str, Any], report_b: dict[str, Any]) -> dict[str, Any]:
    corpus_a = report_a.get("training_corpus_sha256")
    corpus_b = report_b.get("training_corpus_sha256")
    if not corpus_a or corpus_a != corpus_b:
        raise ValueError("A/B candidates must prove the same training corpus SHA-256")

    eval_a = report_a.get("evaluation_corpus", {}).get("sha256")
    eval_b = report_b.get("evaluation_corpus", {}).get("sha256")
    if not eval_a or eval_a != eval_b:
        raise ValueError("A/B candidates must use the same evaluation corpus SHA-256")

    domains_a = set(report_a["domains"])
    domains_b = set(report_b["domains"])
    if domains_a != domains_b:
        raise ValueError("A/B candidate domain sets differ")

    def delta(metric: str, left: dict[str, Any], right: dict[str, Any]) -> float:
        return float(right[metric]) - float(left[metric])

    domain_deltas: dict[str, Any] = {}
    for domain in sorted(domains_a):
        left = report_a["domains"][domain]
        right = report_b["domains"][domain]
        domain_deltas[domain] = {
            "tokens_per_word_b_minus_a": delta("tokens_per_word", left, right),
            "bytes_per_token_b_minus_a": delta("bytes_per_token", left, right),
            "characters_per_token_b_minus_a": delta("characters_per_token", left, right),
            "byte_fallback_rate_b_minus_a": delta("byte_fallback_rate", left, right),
            "roundtrip_failures_a": int(left["roundtrip_failures"]),
            "roundtrip_failures_b": int(right["roundtrip_failures"]),
        }

    return {
        "schema_version": "1.0.0",
        "model_line": "quantum-1-echelon",
        "training_corpus_sha256": corpus_a,
        "evaluation_corpus_sha256": eval_a,
        "candidate_a_model_sha256": report_a["model_sha256"],
        "candidate_b_model_sha256": report_b["model_sha256"],
        "overall": {
            "tokens_per_word_b_minus_a": delta(
                "tokens_per_word", report_a["overall"], report_b["overall"]
            ),
            "bytes_per_token_b_minus_a": delta(
                "bytes_per_token", report_a["overall"], report_b["overall"]
            ),
            "characters_per_token_b_minus_a": delta(
                "characters_per_token", report_a["overall"], report_b["overall"]
            ),
            "byte_fallback_rate_b_minus_a": delta(
                "byte_fallback_rate", report_a["overall"], report_b["overall"]
            ),
            "roundtrip_failures_a": int(report_a["overall"]["roundtrip_failures"]),
            "roundtrip_failures_b": int(report_b["overall"]["roundtrip_failures"]),
        },
        "domains": domain_deltas,
        "decision_status": "maintainer_review_required",
        "decision_rule": (
            "Prefer 32K unless 48K shows a material, repeatable efficiency or coverage "
            "advantage without domain regressions or round-trip failures."
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantum 1 Echelon tokenizer A/B tooling.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--corpus", type=Path, required=True)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--eval-jsonl", type=Path, required=True)
    evaluate.add_argument("--tokenizer-manifest", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)

    compare = sub.add_parser("compare")
    compare.add_argument("--a", type=Path, required=True)
    compare.add_argument("--b", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "train":
        result = train_candidate(args.config, args.corpus)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        result = evaluate_candidate(
            args.model,
            args.eval_jsonl,
            tokenizer_manifest=args.tokenizer_manifest,
        )
        _write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        left = json.loads(args.a.read_text(encoding="utf-8"))
        right = json.loads(args.b.read_text(encoding="utf-8"))
        result = compare_reports(left, right)
        _write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
