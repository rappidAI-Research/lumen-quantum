"""Generiert Text mit einem lokalen quantum-1-base Checkpoint."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

try:
    from .generate import generate_text, load_model_state_dict
    from .inspect_model_size import load_quantum_tokenizer_info, load_yaml_config
    from .train_tokenizer import load_fast_tokenizer
except ImportError:
    from generate import generate_text, load_model_state_dict
    from inspect_model_size import load_quantum_tokenizer_info, load_yaml_config
    from train_tokenizer import load_fast_tokenizer


LOGGER = logging.getLogger("lumen.generate_quantum")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


def load_quantum_model_from_checkpoint(
    checkpoint_dir: str | Path, device: str | None = None
) -> LlamaForCausalLM:
    checkpoint = Path(checkpoint_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint}")
    config_file = checkpoint / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Modell-Config nicht gefunden: {config_file}")
    model_config = LlamaConfig.from_json_file(str(config_file))
    model = LlamaForCausalLM(model_config)
    load_quantum_weights(model, checkpoint)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(resolved_device)
    model.eval()
    return model


def load_quantum_weights(model: LlamaForCausalLM, checkpoint_dir: str | Path) -> None:
    """Laedt lokale quantum-Gewichte ohne from_pretrained.

    Bei tie_word_embeddings=true speichert Transformers die geteilte
    lm_head.weight nicht doppelt. Das ist erwartet und fuer GGUF/HF korrekt.
    """

    state_dict = load_model_state_dict(checkpoint_dir)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"lm_head.weight"} if bool(model.config.tie_word_embeddings) else set()
    remaining_missing = [key for key in missing if key not in allowed_missing]
    if unexpected or remaining_missing:
        raise RuntimeError(
            "Lokaler quantum-Checkpoint passt nicht zur Modellarchitektur: "
            f"missing={remaining_missing}, unexpected={list(unexpected)}"
        )
    if bool(model.config.tie_word_embeddings):
        model.tie_weights()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generiert Text mit quantum-1-base.")
    parser.add_argument("--config", default="configs/quantum_1_base_pilot.yaml")
    parser.add_argument("--checkpoint", help="Lokaler Checkpoint-Ordner. Default aus Config.")
    parser.add_argument(
        "--tokenizer-dir", help="Lokaler Tokenizer-Ordner. Default aus Config oder Checkpoint."
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    checkpoint = Path(args.checkpoint or config["generation"]["checkpoint_dir"])
    if args.tokenizer_dir:
        tokenizer_dir = Path(args.tokenizer_dir)
    elif (checkpoint / "tokenizer" / "tokenizer.model").exists():
        tokenizer_dir = checkpoint / "tokenizer"
    else:
        tokenizer_dir = load_quantum_tokenizer_info(config).tokenizer_dir

    tokenizer = load_fast_tokenizer(tokenizer_dir)
    model = load_quantum_model_from_checkpoint(checkpoint, args.device)
    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=int(args.max_new_tokens or config["generation"]["max_new_tokens"]),
        temperature=float(args.temperature or config["generation"]["temperature"]),
        top_p=float(args.top_p or config["generation"]["top_p"]),
    )
    print(text)


if __name__ == "__main__":
    main()
