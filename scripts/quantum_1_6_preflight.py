"""Preflight-Checks fuer das Continued Pretraining von quantum-1.6-pilot.

Dieses Modul enthaelt bewusst KEINE torch-/transformers-Abhaengigkeit, damit
Konfigurations-, Pfad- und Tokenizer-Kompatibilitaetspruefungen auch ohne
installierten ML-Stack laufen (z. B. in CI oder lokal auf Windows).

Geprueft wird vor allem:
- Struktur/Werte der Trainingskonfiguration
- weights-only Initialisierung ist korrekt konfiguriert (kein Optimizer-/Scheduler-/
  Schrittzustand wird uebernommen)
- der eingefrorene Tokenizer ist byte-identisch zum Tokenizer des Basismodells
- Ausgabe erfolgt ausschliesslich nach models/quantum-1.6-pilot/
- die neuen Datenpfade ueberschreiben die bisherige quantum-1-pilot-Datenbasis nicht
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable

import yaml


LOGGER = logging.getLogger("lumen.quantum_1_6_preflight")

EXPECTED_PARAMETER_COUNT = 49295872
EXPECTED_MODEL_NAME = "quantum-1.6-pilot"

# Korrekter, finaler Tokenizer. Er ist byte-identisch zum Tokenizer im Basismodell
# (models/quantum-1-base/final) und traegt exakt diesen SHA256.
FROZEN_TOKENIZER_DIR = "tokenizer/quantum-1"
EXPECTED_TOKENIZER_SHA256 = "be99b72377f3cb2ce1c875103d0324a2001ee5543a49e7c8fabfc1e384b1b6f6"

# Falscher, inkompatibler alter Pilot-Tokenizer. Darf fuer quantum-1.6-pilot
# niemals verwendet werden (abweichender SHA256, passt nicht zu den Basisgewichten).
INCOMPATIBLE_TOKENIZER_DIR = "tokenizer/quantum-1-pilot"
INCOMPATIBLE_TOKENIZER_SHA256 = "33017b41667f3ac30a60ee383f9018494b4c2c382ab2e83c7d0d219cd7c4c140"

OUTPUT_DIR_PREFIX = "models/quantum-1.6-pilot"
PROTECTED_DATA_DIRS = (
    "data/quantum/cleaned",
    "data/quantum/tokenized/pilot",
    "data/quantum/raw",
)


class PreflightError(RuntimeError):
    """Wird ausgeloest, wenn eine Preflight-Bedingung verletzt ist."""


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_yaml_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_posix(path: str | Path) -> str:
    return Path(path).as_posix().rstrip("/")


def validate_train_config(config: dict) -> list[str]:
    """Prueft Struktur und Werte der Trainingskonfiguration. Gibt Hinweise zurueck."""

    notes: list[str] = []
    project = config.get("project", {})
    if project.get("model_name") != EXPECTED_MODEL_NAME:
        raise PreflightError(
            f"project.model_name muss '{EXPECTED_MODEL_NAME}' sein, ist: {project.get('model_name')!r}."
        )

    init = config.get("init")
    if not isinstance(init, dict):
        raise PreflightError("Abschnitt 'init' fehlt in der Trainingskonfiguration.")
    if not init.get("from_model"):
        raise PreflightError("init.from_model muss auf das finale Basismodell zeigen.")
    if not bool(init.get("weights_only", False)):
        raise PreflightError("init.weights_only muss true sein (nur Gewichte laden).")
    for flag in ("reset_optimizer", "reset_scheduler", "reset_global_step"):
        if not bool(init.get(flag, False)):
            raise PreflightError(
                f"init.{flag} muss true sein: alter Optimizer-/Scheduler-/Schrittzustand darf NICHT uebernommen werden."
            )

    training = config.get("training", {})
    output_dir = _as_posix(training.get("output_dir", ""))
    if not output_dir.startswith(OUTPUT_DIR_PREFIX):
        raise PreflightError(
            f"training.output_dir muss unter '{OUTPUT_DIR_PREFIX}' liegen, ist: {output_dir!r}."
        )

    max_steps = int(training.get("max_steps", 0))
    if max_steps <= 0:
        raise PreflightError("training.max_steps muss fuer das Volltraining groesser als 0 sein.")
    warmup_steps = int(training.get("warmup_steps", 0))
    if warmup_steps <= 0 or warmup_steps >= max_steps:
        raise PreflightError("training.warmup_steps muss > 0 und < max_steps sein.")
    learning_rate = float(training.get("learning_rate", 0.0))
    if not 0.0 < learning_rate <= 5e-4:
        raise PreflightError(
            f"training.learning_rate={learning_rate} unplausibel fuer vorsichtiges Continued Pretraining (erwartet ~1e-4)."
        )
    if float(training.get("max_grad_norm", 0.0)) <= 0:
        raise PreflightError("training.max_grad_norm muss groesser als 0 sein.")
    for key in ("save_steps", "eval_steps"):
        if int(training.get(key, 0)) <= 0:
            raise PreflightError(f"training.{key} muss groesser als 0 sein.")
    if int(training["save_steps"]) > 1000:
        notes.append("Hinweis: save_steps > 1000, Vorgabe war mindestens alle 1.000 Schritte.")
    if int(training["eval_steps"]) > 1000:
        notes.append("Hinweis: eval_steps > 1000, Vorgabe war mindestens alle 1.000 Schritte.")

    # Effektive Batchgroesse in Tokens (Single-GPU) pruefen.
    block_size = int(config.get("data", {}).get("block_size", 512))
    effective_tokens = (
        int(training.get("batch_size", 0))
        * int(training.get("gradient_accumulation_steps", 0))
        * block_size
    )
    notes.append(f"Effektive Batchgroesse (Single-GPU): {effective_tokens} Tokens/Schritt.")

    model = config.get("model", {})
    if int(model.get("parameter_count_expected", 0)) != EXPECTED_PARAMETER_COUNT:
        raise PreflightError(
            f"model.parameter_count_expected muss {EXPECTED_PARAMETER_COUNT} sein."
        )
    if not bool(model.get("tie_word_embeddings", False)):
        raise PreflightError("model.tie_word_embeddings muss true sein (wie quantum-1-pilot).")
    return notes


def check_output_isolation(config: dict, data_config: dict | None = None) -> None:
    """Stellt sicher, dass nichts Bestehendes ueberschrieben wird."""

    output_dir = _as_posix(config["training"]["output_dir"])
    init_from = _as_posix(config["init"]["from_model"])
    if output_dir == init_from or output_dir.startswith(init_from + "/"):
        raise PreflightError("training.output_dir darf nicht in init.from_model liegen.")
    if init_from.startswith(output_dir + "/") or init_from == output_dir:
        raise PreflightError("init.from_model darf nicht im Output-Verzeichnis liegen.")
    if "quantum-1-base" in output_dir or "quantum-1-pilot" in output_dir:
        raise PreflightError(
            f"training.output_dir {output_dir!r} darf nicht auf bestehende quantum-1-Artefakte zeigen."
        )

    if data_config is not None:
        new_dirs = {_as_posix(p) for p in data_config.get("paths", {}).values()}
        new_dirs.add(_as_posix(data_config.get("output", {}).get("dir", "")))
        for protected in PROTECTED_DATA_DIRS:
            if _as_posix(protected) in new_dirs:
                raise PreflightError(
                    f"Neue Datenpfade duerfen die bestehende Datenbasis nicht ueberschreiben: {protected}."
                )


def check_tokenizer_compatibility(config: dict) -> dict:
    """Prueft, dass der korrekte finale Tokenizer verwendet wird.

    quantum-1.6-pilot MUSS tokenizer/quantum-1 verwenden. Dieser ist byte-identisch
    zum Tokenizer im Basismodell (models/quantum-1-base/final). Explizit geprueft wird:
    - tokenizer/quantum-1/tokenizer.model
    - models/quantum-1-base/final/tokenizer.model
    - beide SHA256 identisch
    - der Hash ist der erwartete Wert (be99b723...)
    Der falsche, inkompatible alte Pilot-Tokenizer (tokenizer/quantum-1-pilot,
    Hash 33017b41...) wird explizit abgelehnt.
    """

    tokenizer_dir = Path(config["tokenizer"]["dir"])
    if _as_posix(tokenizer_dir) == _as_posix(INCOMPATIBLE_TOKENIZER_DIR):
        raise PreflightError(
            f"{INCOMPATIBLE_TOKENIZER_DIR} ist der falsche, inkompatible alte Pilot-Tokenizer "
            f"(SHA256 {INCOMPATIBLE_TOKENIZER_SHA256}). quantum-1.6-pilot muss {FROZEN_TOKENIZER_DIR} verwenden."
        )
    if _as_posix(tokenizer_dir) != _as_posix(FROZEN_TOKENIZER_DIR):
        raise PreflightError(
            f"tokenizer.dir muss der korrekte finale Tokenizer {FROZEN_TOKENIZER_DIR!r} sein, ist: {_as_posix(tokenizer_dir)!r}."
        )
    base_tokenizer_dir = Path(config["init"].get("base_model_tokenizer_dir", config["init"]["from_model"]))

    frozen_model = tokenizer_dir / "tokenizer.model"
    base_model_tok = base_tokenizer_dir / "tokenizer.model"
    for path in (frozen_model, base_model_tok):
        if not path.exists():
            raise FileNotFoundError(f"tokenizer.model nicht gefunden: {path}")

    frozen_hash = sha256_file(frozen_model)
    base_hash = sha256_file(base_model_tok)

    if frozen_hash == INCOMPATIBLE_TOKENIZER_SHA256:
        raise PreflightError(
            "Der konfigurierte Tokenizer ist der falsche, inkompatible alte Pilot-Tokenizer "
            f"(SHA256 {INCOMPATIBLE_TOKENIZER_SHA256}). Erwartet wird {EXPECTED_TOKENIZER_SHA256}."
        )
    if frozen_hash != base_hash:
        raise PreflightError(
            "Tokenizer-Inkompatibilitaet: tokenizer.model des Tokenizers "
            f"({frozen_hash}) unterscheidet sich vom Tokenizer des Basismodells ({base_hash})."
        )
    if frozen_hash != EXPECTED_TOKENIZER_SHA256:
        raise PreflightError(
            f"Tokenizer-Hash {frozen_hash} entspricht nicht dem erwarteten Wert {EXPECTED_TOKENIZER_SHA256}."
        )

    # Manifest defensiv lesen (Name/Vokabular sind hilfreich, aber der SHA256 ist maßgeblich).
    tokenizer_name = None
    vocab_size = None
    manifest_path = tokenizer_dir / config["tokenizer"].get("manifest_file", "tokenizer_manifest.json")
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        tokenizer_name = manifest.get("tokenizer_name")
        vocab_size = int(manifest.get("actual_vocab_size", 0)) or None
        if tokenizer_name == "quantum-1-pilot":
            raise PreflightError(
                "Tokenizer-Manifest weist 'quantum-1-pilot' aus — das ist der falsche, inkompatible Tokenizer."
            )

    return {
        "tokenizer_dir": _as_posix(tokenizer_dir),
        "tokenizer_name": tokenizer_name,
        "vocab_size": vocab_size,
        "tokenizer_model_sha256": frozen_hash,
        "base_model_tokenizer_sha256": base_hash,
        "expected_tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "identical_to_base_model": frozen_hash == base_hash,
        "matches_expected_hash": frozen_hash == EXPECTED_TOKENIZER_SHA256,
    }


def check_base_model_present(config: dict) -> dict:
    """Stellt sicher, dass das finale Basismodell vollstaendig vorhanden ist."""

    base_dir = Path(config["init"]["from_model"])
    if not base_dir.exists():
        raise FileNotFoundError(f"Basismodell-Ordner nicht gefunden: {base_dir}")
    config_json = base_dir / "config.json"
    has_weights = (base_dir / "model.safetensors").exists() or (base_dir / "pytorch_model.bin").exists()
    if not config_json.exists():
        raise FileNotFoundError(f"config.json des Basismodells fehlt: {config_json}")
    if not has_weights:
        raise FileNotFoundError(
            f"Basismodell-Gewichte (model.safetensors/pytorch_model.bin) fehlen in: {base_dir}"
        )
    return {
        "base_model_dir": _as_posix(base_dir),
        "config_json": str(config_json),
        "weights_file": "model.safetensors" if (base_dir / "model.safetensors").exists() else "pytorch_model.bin",
    }


def run_preflight(train_config_path: str | Path, data_config_path: str | Path | None = None) -> dict:
    """Fuehrt alle torch-freien Preflight-Checks aus und liefert einen Report."""

    config = load_yaml_config(train_config_path)
    data_config = load_yaml_config(data_config_path) if data_config_path else None

    notes = validate_train_config(config)
    check_output_isolation(config, data_config)
    base_report = check_base_model_present(config)
    tokenizer_report = check_tokenizer_compatibility(config)

    report = {
        "train_config": str(train_config_path),
        "data_config": str(data_config_path) if data_config_path else None,
        "model_name": config["project"]["model_name"],
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "weights_only_init": True,
        "reset_optimizer_scheduler_step": True,
        "base_model": base_report,
        "tokenizer": tokenizer_report,
        "notes": notes,
        "torch_free": True,
    }
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Torch-freie Preflight-Checks fuer quantum-1.6-pilot.")
    parser.add_argument("--config", default="configs/quantum_1_6_pilot_train.yaml")
    parser.add_argument("--data-config", default="configs/quantum_1_6_pilot_data.yaml")
    parser.add_argument("--json", action="store_true", help="Report als JSON ausgeben.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    report = run_preflight(args.config, args.data_config)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    LOGGER.info("Preflight OK fuer %s", report["model_name"])
    LOGGER.info("Tokenizer identisch zum Basismodell: %s", report["tokenizer"]["identical_to_base_model"])
    LOGGER.info("Basismodell-Gewichte: %s/%s", report["base_model"]["base_model_dir"], report["base_model"]["weights_file"])
    for note in report["notes"]:
        LOGGER.info(note)


if __name__ == "__main__":
    main()
