"""Validiert und friert den finalen quantum-1 Tokenizer ein.

Das Skript erweitert die generische quantum-Tokenizer-Validierung um finale
Pfad-, Split- und Freeze-Pruefungen. Es trainiert keinen Tokenizer und laedt
keine vortrainierten Tokenizer oder Modellgewichte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable

import yaml

try:
    import sentencepiece as spm
except ImportError as exc:  # pragma: no cover - depends on local environment.
    raise ImportError("sentencepiece ist erforderlich. Installiere: pip install -r requirements.txt") from exc

try:
    from .validate_quantum_tokenizer import validate_quantum_tokenizer
except ImportError:
    from validate_quantum_tokenizer import validate_quantum_tokenizer


EXPECTED_SPECIAL_TOKEN_IDS = {
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    "<pad>": 3,
    "<|system|>": 4,
    "<|user|>": 5,
    "<|assistant|>": 6,
}

REQUIRED_GGUF_FILES = [
    "tokenizer.model",
    "tokenizer.vocab",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer_manifest.json",
]

FREEZE_MARKER_FILE = "FINAL_FROZEN"


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_parts(path: str | Path) -> tuple[str, ...]:
    return PurePosixPath(str(path).replace("\\", "/")).parts


def contains_sequence(path: str | Path, expected: tuple[str, ...]) -> bool:
    parts = path_parts(path)
    width = len(expected)
    return any(parts[index : index + width] == expected for index in range(len(parts) - width + 1))


def reject_pilot_path(path: str | Path, label: str) -> None:
    normalized = str(path).replace("\\", "/").lower()
    if "pilot" in normalized:
        raise ValueError(f"{label} darf keinen Pilotpfad verwenden: {path}")


def require_final_tokenizer_path(path: str | Path, label: str = "tokenizer_dir") -> None:
    reject_pilot_path(path, label)
    if not contains_sequence(path, ("tokenizer", "quantum-1")):
        raise ValueError(f"{label} muss unter tokenizer/quantum-1 liegen: {path}")


def require_train_only_file(path: str | Path) -> None:
    normalized = str(path).replace("\\", "/").lower()
    if not normalized.endswith("/train.jsonl") and not normalized.endswith("train.jsonl"):
        raise ValueError(f"Tokenizer-Training muss train.jsonl verwenden: {path}")
    if "validation.jsonl" in normalized or "test.jsonl" in normalized:
        raise ValueError(f"Validation/Test duerfen nicht fuer Tokenizer-Training genutzt werden: {path}")
    if not contains_sequence(path, ("data", "quantum", "final")):
        raise ValueError(f"Finaler Tokenizer muss auf finalen Train-Daten trainiert werden: {path}")


def validate_special_token_order(sp: spm.SentencePieceProcessor, manifest: dict) -> dict[str, int]:
    actual = {token: int(sp.piece_to_id(token)) for token in EXPECTED_SPECIAL_TOKEN_IDS}
    for token, expected_id in EXPECTED_SPECIAL_TOKEN_IDS.items():
        if actual[token] != expected_id:
            raise ValueError(f"{token} muss Token-ID {expected_id} haben, hat aber {actual[token]}.")
    if list(manifest.get("expected_special_token_order", EXPECTED_SPECIAL_TOKEN_IDS.keys())) != list(
        EXPECTED_SPECIAL_TOKEN_IDS.keys()
    ):
        raise ValueError("tokenizer_manifest.json dokumentiert nicht die stabile Special-Token-Reihenfolge.")
    for token, expected_id in EXPECTED_SPECIAL_TOKEN_IDS.items():
        manifest_id = int(manifest.get("token_ids", {}).get(token, -1))
        if manifest_id != expected_id:
            raise ValueError(f"Manifest-ID fuer {token} muss {expected_id} sein, ist aber {manifest_id}.")
    return actual


def validate_training_data_hash(manifest: dict) -> None:
    train_file = Path(manifest["training_file"])
    require_train_only_file(train_file)
    if not train_file.exists():
        raise FileNotFoundError(f"Tokenizer-Trainingsdatei aus Manifest fehlt: {train_file}")
    actual_hash = sha256_file(train_file)
    manifest_hash = manifest.get("training_data_sha256")
    if actual_hash != manifest_hash:
        raise ValueError(
            f"Trainingsdaten-Hash stimmt nicht: manifest={manifest_hash}, aktuell={actual_hash}."
        )


def validate_model_vocab_match(tokenizer_vocab_size: int, train_config_path: str | Path | None) -> None:
    if not train_config_path:
        return
    config_file = Path(train_config_path)
    if not config_file.exists():
        return
    config = load_yaml(config_file)
    require_final_tokenizer_path(config["tokenizer"]["dir"], "train_config.tokenizer.dir")
    reject_pilot_path(config["training"]["output_dir"], "training.output_dir")
    if not contains_sequence(config["training"]["output_dir"], ("models", "quantum-1-base")):
        raise ValueError("Finales Training muss unter models/quantum-1-base ausgeben.")

    configured_vocab_size = config["model"].get("vocab_size", "auto")
    if configured_vocab_size != "auto" and int(configured_vocab_size) != int(tokenizer_vocab_size):
        raise ValueError(
            f"Modell-vocab_size={configured_vocab_size} passt nicht zum Tokenizer={tokenizer_vocab_size}."
        )


def write_freeze_files(tokenizer_dir: Path, config_path: str | Path, report: dict) -> None:
    freeze_manifest = {
        "status": "final_frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path),
        "tokenizer_dir": str(tokenizer_dir),
        "tokenizer_model_sha256": sha256_file(tokenizer_dir / "tokenizer.model"),
        "tokenizer_manifest_sha256": sha256_file(tokenizer_dir / "tokenizer_manifest.json"),
        "validation_report_sha256": sha256_file(tokenizer_dir / "validation_report.json"),
        "token_ids": report["token_ids"],
        "note": "Finaler quantum-1 Tokenizer. Nicht ueberschreiben; fuer neue Version neuen Ordner verwenden.",
    }
    (tokenizer_dir / "freeze_manifest.json").write_text(
        json.dumps(freeze_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (tokenizer_dir / FREEZE_MARKER_FILE).write_text(
        json.dumps(freeze_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_final_tokenizer(
    config_path: str | Path = "configs/quantum_1_final_tokenizer.yaml",
    train_config_path: str | Path | None = "configs/quantum_1_final_train.yaml",
    freeze: bool = True,
    allow_missing: bool = False,
) -> dict:
    config = load_yaml(config_path)
    tokenizer_dir = Path(config["tokenizer"]["output_dir"])
    require_final_tokenizer_path(tokenizer_dir)
    require_train_only_file(config["data"]["train_file"])
    validate_model_vocab_match(int(config["tokenizer"]["vocab_size"]), train_config_path)

    missing = [filename for filename in REQUIRED_GGUF_FILES if not (tokenizer_dir / filename).exists()]
    if missing:
        if allow_missing:
            return {
                "final_validation": {
                    "ok": True,
                    "validated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config_only": True,
                    "missing": missing,
                    "frozen": False,
                    "uses_pretrained_tokenizer": False,
                },
                "tokenizer_dir": str(tokenizer_dir),
                "expected_vocab_size": int(config["tokenizer"]["vocab_size"]),
            }
        raise FileNotFoundError(f"Finale Tokenizerdateien fehlen in {tokenizer_dir}: {', '.join(missing)}")

    base_report = validate_quantum_tokenizer(config_path)
    manifest = read_json(tokenizer_dir / "tokenizer_manifest.json")
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_dir / "tokenizer.model"))
    expected_vocab_size = int(config["tokenizer"]["vocab_size"])
    actual_vocab_size = int(sp.get_piece_size())
    if actual_vocab_size != expected_vocab_size:
        raise ValueError(f"Tokenizer-vocab_size={actual_vocab_size}, erwartet {expected_vocab_size}.")
    if int(manifest.get("requested_vocab_size", -1)) != expected_vocab_size:
        raise ValueError("tokenizer_manifest.json requested_vocab_size passt nicht zur finalen Config.")
    if int(manifest.get("future_llama_vocab_size", -1)) != expected_vocab_size:
        raise ValueError("tokenizer_manifest.json future_llama_vocab_size passt nicht zur finalen Config.")

    token_ids = validate_special_token_order(sp, manifest)
    validate_training_data_hash(manifest)
    validate_model_vocab_match(actual_vocab_size, train_config_path)

    report = {
        **base_report,
        "final_validation": {
            "ok": True,
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_only": True,
            "gguf_ready": True,
            "frozen": bool(freeze),
            "uses_pretrained_tokenizer": False,
        },
        "token_ids": token_ids,
        "gguf_files": {filename: sha256_file(tokenizer_dir / filename) for filename in REQUIRED_GGUF_FILES},
    }
    (tokenizer_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if freeze:
        write_freeze_files(tokenizer_dir, config_path, report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validiert und friert tokenizer/quantum-1 ein.")
    parser.add_argument("--config", default="configs/quantum_1_final_tokenizer.yaml")
    parser.add_argument("--train-config", default="configs/quantum_1_final_train.yaml")
    parser.add_argument("--no-freeze", action="store_true", help="Nur validieren, keinen FINAL_FROZEN Marker schreiben.")
    parser.add_argument("--allow-missing", action="store_true", help="Nur Config/Pfade pruefen, fehlende Tokenizerdateien erlauben.")
    parser.add_argument("--json", action="store_true", help="Validierungsreport als JSON ausgeben.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = validate_final_tokenizer(
        config_path=args.config,
        train_config_path=args.train_config,
        freeze=not args.no_freeze,
        allow_missing=args.allow_missing,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        if report.get("final_validation", {}).get("config_only"):
            print("Finaler Tokenizer: Config/Pfade OK; Tokenizerdateien noch nicht vorhanden.")
        else:
            print("Finaler Tokenizer validiert und eingefroren." if not args.no_freeze else "Finaler Tokenizer validiert.")


if __name__ == "__main__":
    main()
