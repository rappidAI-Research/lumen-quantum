"""Einfache lokale Evaluation fuer quantum-1-base Checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    from .evaluate import read_prompts
    from .generate import generate_text
    from .generate_quantum import load_quantum_model_from_checkpoint
    from .inspect_model_size import load_quantum_tokenizer_info, load_yaml_config
    from .train_tokenizer import load_fast_tokenizer
except ImportError:
    from evaluate import read_prompts
    from generate import generate_text
    from generate_quantum import load_quantum_model_from_checkpoint
    from inspect_model_size import load_quantum_tokenizer_info, load_yaml_config
    from train_tokenizer import load_fast_tokenizer


LOGGER = logging.getLogger("lumen.evaluate_quantum")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def run_quantum_evaluation(
    config_path: str | Path,
    checkpoint: str | Path | None = None,
    eval_file: str | Path | None = None,
    output_file: str | Path | None = None,
    device: str | None = None,
) -> Path:
    config = load_yaml_config(config_path)
    checkpoint_path = Path(checkpoint or config["generation"]["checkpoint_dir"])
    resolved_eval_file = Path(eval_file or config["evaluation"]["eval_file"])
    resolved_output_file = Path(output_file or config["evaluation"]["output_file"])

    if (checkpoint_path / "tokenizer" / "tokenizer.model").exists():
        tokenizer_dir = checkpoint_path / "tokenizer"
    else:
        tokenizer_dir = load_quantum_tokenizer_info(config).tokenizer_dir

    tokenizer = load_fast_tokenizer(tokenizer_dir)
    model = load_quantum_model_from_checkpoint(checkpoint_path, device)
    prompts = read_prompts(resolved_eval_file)
    resolved_output_file.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_file.open("w", encoding="utf-8") as handle:
        for index, prompt in enumerate(prompts, start=1):
            generated = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=int(config["evaluation"]["max_new_tokens"]),
                temperature=float(config["evaluation"]["temperature"]),
                top_p=float(config["evaluation"]["top_p"]),
            )
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "checkpoint": str(checkpoint_path),
                        "prompt": prompt,
                        "generated_text": generated,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    LOGGER.info("quantum Evaluation gespeichert: %s", resolved_output_file)
    return resolved_output_file


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuehrt eine einfache quantum-1-base Evaluation aus.")
    parser.add_argument("--config", default="configs/quantum_1_base_pilot.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--eval-file")
    parser.add_argument("--output-file")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    run_quantum_evaluation(args.config, args.checkpoint, args.eval_file, args.output_file, args.device)


if __name__ == "__main__":
    main()
