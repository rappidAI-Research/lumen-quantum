"""Generiert Text mit einem lokalen Lumen-Smoke-Checkpoint."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import torch
from transformers import LlamaConfig, LlamaForCausalLM

try:
    from .train_tokenizer import load_fast_tokenizer
except ImportError:
    from train_tokenizer import load_fast_tokenizer


LOGGER = logging.getLogger("lumen.generate")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_model_state_dict(checkpoint_dir: str | Path) -> dict:
    """Laedt die reinen Gewichte eines lokalen Checkpoints.

    Bevorzugt model.safetensors (sicher, GGUF-freundlich), faellt auf
    pytorch_model.bin zurueck. Es werden nur lokale, selbst erzeugte Gewichte
    geladen; niemals from_pretrained.
    """

    checkpoint = Path(checkpoint_dir)
    safetensors_file = checkpoint / "model.safetensors"
    if safetensors_file.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_file))
    bin_file = checkpoint / "pytorch_model.bin"
    if bin_file.exists():
        return torch.load(bin_file, map_location="cpu", weights_only=True)
    raise FileNotFoundError(
        f"Kein model.safetensors oder pytorch_model.bin in {checkpoint} gefunden."
    )


def load_model_from_checkpoint(checkpoint_dir: str | Path, device: str | None = None) -> LlamaForCausalLM:
    checkpoint = Path(checkpoint_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint}")
    config_file = checkpoint / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Modell-Config nicht gefunden: {config_file}")

    model_config = LlamaConfig.from_json_file(str(config_file))
    model = LlamaForCausalLM(model_config)
    model.load_state_dict(load_model_state_dict(checkpoint))

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(resolved_device)
    model.eval()
    LOGGER.info("Lokaler Checkpoint geladen: %s auf %s", checkpoint, resolved_device)
    return model


@torch.no_grad()
def generate_text(
    model: LlamaForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    if not prompt.strip():
        raise ValueError("Prompt darf nicht leer sein.")
    if temperature <= 0:
        raise ValueError("temperature muss groesser als 0 sein.")
    if not 0 < top_p <= 1:
        raise ValueError("top_p muss zwischen 0 und 1 liegen.")

    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)

    if tokenizer.bos_token_id is not None:
        bos = torch.tensor([[tokenizer.bos_token_id]], dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([bos, input_ids], dim=1)

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generiert Text mit einem lokalen Lumen-Checkpoint.")
    parser.add_argument("--checkpoint", default="models/smoke/final", help="Lokaler Checkpoint-Ordner.")
    parser.add_argument("--tokenizer-dir", default="tokenizer/smoke", help="Lokaler Tokenizer-Ordner.")
    parser.add_argument("--prompt", required=True, help="Prompt fuer die Generierung.")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Maximal neu zu generierende Tokens.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling-Temperatur.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus-Sampling top-p.")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Optionales Zielgeraet.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    checkpoint = Path(args.checkpoint)
    tokenizer_dir = Path(args.tokenizer_dir)
    if (checkpoint / "tokenizer" / "tokenizer.json").exists() and args.tokenizer_dir == "tokenizer/smoke":
        tokenizer_dir = checkpoint / "tokenizer"

    tokenizer = load_fast_tokenizer(tokenizer_dir)
    model = load_model_from_checkpoint(checkpoint, args.device)
    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(text)


if __name__ == "__main__":
    main()
