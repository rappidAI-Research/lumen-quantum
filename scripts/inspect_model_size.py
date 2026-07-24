"""Prueft die Groesse der quantum-1-base Pilot-Architektur.

Das Modell wird ausschliesslich direkt aus LlamaConfig initialisiert. Es werden
keine vortrainierten Modellgewichte und kein from_pretrained fuer Modelle
verwendet.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from transformers import LlamaConfig, LlamaForCausalLM

try:
    import sentencepiece as spm
except ImportError as exc:  # pragma: no cover - depends on local environment.
    raise ImportError(
        "sentencepiece ist fuer die lokale Tokenizer-Pruefung erforderlich. "
        "Installiere zuerst: pip install -r requirements.txt"
    ) from exc


LOGGER = logging.getLogger("lumen.inspect_model_size")

EXPECTED_BASE_TOKENS = {
    "unk_token": ("<unk>", 0),
    "bos_token": ("<s>", 1),
    "eos_token": ("</s>", 2),
    "pad_token": ("<pad>", 3),
}

REQUIRED_TOKENIZER_FILES = [
    "tokenizer.model",
    "tokenizer.vocab",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer_manifest.json",
]


@dataclass(frozen=True)
class QuantumTokenizerInfo:
    tokenizer_dir: Path
    tokenizer_model: Path
    vocab_size: int
    unk_token_id: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    tokenizer_class: str
    token_ids: dict[str, int]
    manifest: dict


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


def read_json(path: str | Path) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON-Datei nicht gefunden: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_tokenizer_dir(config: dict) -> Path:
    tokenizer_dir = Path(config["tokenizer"]["dir"])
    if not tokenizer_dir.exists():
        raise FileNotFoundError(
            f"Tokenizer-Ordner nicht gefunden: {tokenizer_dir}. "
            "Fuehre zuerst scripts/train_quantum_tokenizer.py aus."
        )
    return tokenizer_dir


def load_quantum_tokenizer_info(config: dict) -> QuantumTokenizerInfo:
    tokenizer_dir = resolve_tokenizer_dir(config)
    missing = [name for name in REQUIRED_TOKENIZER_FILES if not (tokenizer_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Tokenizer-Dateien fehlen in {tokenizer_dir}: {', '.join(missing)}. "
            "Der quantum-1-pilot-Tokenizer muss zuerst vollstaendig erzeugt und validiert werden."
        )

    tokenizer_config = read_json(tokenizer_dir / "tokenizer_config.json")
    special_tokens_map = read_json(tokenizer_dir / "special_tokens_map.json")
    manifest = read_json(
        tokenizer_dir / config["tokenizer"].get("manifest_file", "tokenizer_manifest.json")
    )

    tokenizer_model = tokenizer_dir / "tokenizer.model"
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    vocab_size = int(sp.get_piece_size())

    if tokenizer_config.get("tokenizer_class") != "LlamaTokenizer":
        raise ValueError(
            "tokenizer_config.json muss tokenizer_class='LlamaTokenizer' enthalten, "
            f"gefunden: {tokenizer_config.get('tokenizer_class')!r}."
        )

    token_ids: dict[str, int] = {}
    for key, (piece, expected_id) in EXPECTED_BASE_TOKENS.items():
        if tokenizer_config.get(key) != piece:
            raise ValueError(f"tokenizer_config.json {key} muss {piece!r} sein.")
        if special_tokens_map.get(key) != piece:
            raise ValueError(f"special_tokens_map.json {key} muss {piece!r} sein.")
        actual_id = int(sp.piece_to_id(piece))
        if actual_id != expected_id:
            raise ValueError(f"{piece!r} muss Token-ID {expected_id} haben, hat aber {actual_id}.")
        token_ids[piece] = actual_id

    for token in manifest.get("additional_special_tokens", []):
        token_id = int(sp.piece_to_id(token))
        if token_id < 0 or token_id == int(sp.unk_id()):
            raise ValueError(f"Zusaetzliches Sondertoken fehlt im Tokenizer: {token}")
        token_ids[token] = token_id

    manifest_vocab = int(manifest.get("actual_vocab_size", -1))
    if manifest_vocab != vocab_size:
        raise ValueError(
            f"Tokenizer-Manifest vocab_size={manifest_vocab}, tokenizer.model={vocab_size}."
        )

    return QuantumTokenizerInfo(
        tokenizer_dir=tokenizer_dir,
        tokenizer_model=tokenizer_model,
        vocab_size=vocab_size,
        unk_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        tokenizer_class=str(tokenizer_config["tokenizer_class"]),
        token_ids=token_ids,
        manifest=manifest,
    )


def build_quantum_llama_config(config: dict, tokenizer_info: QuantumTokenizerInfo) -> LlamaConfig:
    model_config = config["model"]
    configured_vocab_size = model_config.get("vocab_size", "auto")
    if configured_vocab_size != "auto" and int(configured_vocab_size) != tokenizer_info.vocab_size:
        raise ValueError(
            f"Modell-vocab_size={configured_vocab_size} passt nicht zum Tokenizer={tokenizer_info.vocab_size}."
        )

    return LlamaConfig(
        vocab_size=tokenizer_info.vocab_size,
        hidden_size=int(model_config["hidden_size"]),
        intermediate_size=int(model_config["intermediate_size"]),
        num_hidden_layers=int(model_config["num_hidden_layers"]),
        num_attention_heads=int(model_config["num_attention_heads"]),
        num_key_value_heads=int(model_config["num_key_value_heads"]),
        max_position_embeddings=int(model_config["max_position_embeddings"]),
        rms_norm_eps=float(model_config.get("rms_norm_eps", 1e-6)),
        rope_theta=float(model_config.get("rope_theta", 10000.0)),
        tie_word_embeddings=bool(model_config.get("tie_word_embeddings", True)),
        attention_dropout=float(model_config.get("attention_dropout", 0.0)),
        initializer_range=float(model_config.get("initializer_range", 0.02)),
        bos_token_id=tokenizer_info.bos_token_id,
        eos_token_id=tokenizer_info.eos_token_id,
        pad_token_id=tokenizer_info.pad_token_id,
        unk_token_id=tokenizer_info.unk_token_id,
    )


def build_quantum_model(llama_config: LlamaConfig) -> LlamaForCausalLM:
    # Direkte Konstruktion aus Config: keine externen oder vortrainierten Gewichte.
    return LlamaForCausalLM(llama_config)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def inspect_model_size(config_path: str | Path) -> dict:
    config = load_yaml_config(config_path)
    set_reproducible_seed(int(config["seed"]))
    tokenizer_info = load_quantum_tokenizer_info(config)
    llama_config = build_quantum_llama_config(config, tokenizer_info)
    model = build_quantum_model(llama_config)
    parameter_count = count_parameters(model)
    min_parameters = int(config["model"]["parameter_count_min"])
    max_parameters = int(config["model"]["parameter_count_max"])

    result = {
        "model_name": config["project"]["model_name"],
        "tokenizer_dir": str(tokenizer_info.tokenizer_dir),
        "vocab_size": tokenizer_info.vocab_size,
        "parameter_count": parameter_count,
        "parameter_count_min": min_parameters,
        "parameter_count_max": max_parameters,
        "within_target_range": min_parameters <= parameter_count <= max_parameters,
        "tie_word_embeddings": bool(llama_config.tie_word_embeddings),
        "max_position_embeddings": int(llama_config.max_position_embeddings),
    }
    if not result["within_target_range"]:
        raise ValueError(
            "Parameterzahl liegt ausserhalb des Zielbereichs: "
            f"{parameter_count:,} Parameter, erwartet {min_parameters:,} bis {max_parameters:,}."
        )
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prueft die quantum-1-base Pilot-Modellgroesse.")
    parser.add_argument("--config", default="configs/quantum_1_base_pilot.yaml")
    parser.add_argument("--json", action="store_true", help="Ausgabe als JSON.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    result = inspect_model_size(args.config)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"Modell: {result['model_name']}")
    print(f"Tokenizer: {result['tokenizer_dir']}")
    print(f"Vocab size: {result['vocab_size']:,}")
    print(f"Parameter: {result['parameter_count']:,}")
    print(
        "Zielbereich: "
        f"{result['parameter_count_min']:,} bis {result['parameter_count_max']:,} "
        f"-> {'OK' if result['within_target_range'] else 'NICHT OK'}"
    )


if __name__ == "__main__":
    main()
