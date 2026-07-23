"""Validiert quantum-1-base lokal ohne grosses Training."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

try:
    from .generate_quantum import load_quantum_weights
    from .inspect_model_size import (
        build_quantum_llama_config,
        build_quantum_model,
        count_parameters,
        inspect_model_size,
        load_quantum_tokenizer_info,
        load_yaml_config,
        set_reproducible_seed,
    )
    from .train_smoke import copy_tokenizer_files_to_model_root
except ImportError:
    from generate_quantum import load_quantum_weights
    from inspect_model_size import (
        build_quantum_llama_config,
        build_quantum_model,
        count_parameters,
        inspect_model_size,
        load_quantum_tokenizer_info,
        load_yaml_config,
        set_reproducible_seed,
    )
    from train_smoke import copy_tokenizer_files_to_model_root


LOGGER = logging.getLogger("lumen.validate_quantum_model")


def save_and_reload_model(model: LlamaForCausalLM, tokenizer_dir: Path) -> LlamaForCausalLM:
    with tempfile.TemporaryDirectory(prefix="lumen_quantum_model_validate_") as temp_name:
        temp_dir = Path(temp_name)
        model.save_pretrained(temp_dir, safe_serialization=True)
        copy_tokenizer_files_to_model_root(tokenizer_dir, temp_dir)

        reloaded_config = LlamaConfig.from_json_file(str(temp_dir / "config.json"))
        reloaded = LlamaForCausalLM(reloaded_config)
        load_quantum_weights(reloaded, temp_dir)
        reloaded.eval()
        return reloaded


@torch.no_grad()
def run_forward_pass(model: LlamaForCausalLM, vocab_size: int) -> dict:
    model.eval()
    input_ids = torch.randint(0, vocab_size, (1, 16), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    if outputs.loss is None or not torch.isfinite(outputs.loss):
        raise ValueError("Forward Pass lieferte keinen endlichen Loss.")
    expected_shape = (1, 16, vocab_size)
    if tuple(outputs.logits.shape) != expected_shape:
        raise ValueError(
            f"Logit-Shape falsch: {tuple(outputs.logits.shape)}, erwartet {expected_shape}."
        )
    return {"loss": float(outputs.loss.item()), "logits_shape": list(outputs.logits.shape)}


def validate_quantum_model(config_path: str | Path) -> dict:
    config = load_yaml_config(config_path)
    set_reproducible_seed(int(config["seed"]))
    tokenizer_info = load_quantum_tokenizer_info(config)
    size_report = inspect_model_size(config_path)
    llama_config = build_quantum_llama_config(config, tokenizer_info)

    if llama_config.vocab_size != tokenizer_info.vocab_size:
        raise ValueError("Tokenizer- und Modell-vocab_size stimmen nicht ueberein.")
    if (
        llama_config.bos_token_id != 1
        or llama_config.eos_token_id != 2
        or llama_config.pad_token_id != 3
    ):
        raise ValueError(
            "BOS/EOS/PAD Token-IDs stimmen nicht mit dem LLaMA-kompatiblen Tokenizer ueberein."
        )
    if getattr(llama_config, "unk_token_id", 0) != 0:
        raise ValueError("UNK Token-ID muss 0 sein.")

    model = build_quantum_model(llama_config)
    forward_report = run_forward_pass(model, tokenizer_info.vocab_size)
    reloaded = save_and_reload_model(model, tokenizer_info.tokenizer_dir)
    reload_report = run_forward_pass(reloaded, tokenizer_info.vocab_size)
    parameter_count = count_parameters(model)

    report = {
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "model_name": config["project"]["model_name"],
        "tokenizer_dir": str(tokenizer_info.tokenizer_dir),
        "vocab_size": tokenizer_info.vocab_size,
        "parameter_count": parameter_count,
        "size_report": size_report,
        "token_ids": tokenizer_info.token_ids,
        "forward_pass": forward_report,
        "save_and_reload": {
            "ok": True,
            "forward_after_reload": reload_report,
            "temporary_artifacts_only": True,
        },
        "uses_pretrained_model_weights": False,
    }
    LOGGER.info("quantum-1-base Validierung erfolgreich: %s Parameter.", f"{parameter_count:,}")
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validiert quantum-1-base ohne grosses Training.")
    parser.add_argument("--config", default="configs/quantum_1_base_pilot.yaml")
    parser.add_argument("--json", action="store_true", help="Ausgabe als JSON.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    args = parse_args(argv)
    report = validate_quantum_model(args.config)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Modell validiert: {report['model_name']}")
        print(f"Parameter: {report['parameter_count']:,}")
        print(f"Tokenizer-Vocab: {report['vocab_size']:,}")
        print("Forward Pass und temporaeres Speichern/Laden: OK")


if __name__ == "__main__":
    main()
