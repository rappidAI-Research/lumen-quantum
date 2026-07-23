"""Einfache Smoke-Evaluation fuer lokale Lumen-Checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

try:
    from .generate import generate_text, load_model_from_checkpoint
    from .train_tokenizer import load_fast_tokenizer
except ImportError:
    from generate import generate_text, load_model_from_checkpoint
    from train_tokenizer import load_fast_tokenizer


LOGGER = logging.getLogger("lumen.evaluate")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def find_eval_file(eval_dir: str | Path) -> Path:
    root = Path(eval_dir)
    if not root.exists():
        raise FileNotFoundError(f"Eval-Ordner nicht gefunden: {root}")
    files = sorted(path for path in root.glob("*.txt") if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"Keine .txt-Eval-Datei in {root} gefunden. Lege z.B. data/evals/prompts.txt an."
        )
    return files[0]


def read_prompts(eval_file: str | Path) -> list[str]:
    path = Path(eval_file)
    if not path.exists():
        raise FileNotFoundError(f"Eval-Datei nicht gefunden: {path}")
    text = path.read_text(encoding="utf-8")
    prompts = [line.strip() for line in text.splitlines() if line.strip()]
    if not prompts:
        prompts = [part.strip() for part in text.split("\n\n") if part.strip()]
    if not prompts:
        raise ValueError(f"Eval-Datei ist leer: {path}")
    return prompts


def run_evaluation(
    checkpoint: str | Path,
    tokenizer_dir: str | Path,
    eval_file: str | Path,
    output_file: str | Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint)
    resolved_tokenizer_dir = Path(tokenizer_dir)
    if (checkpoint_path / "tokenizer" / "tokenizer.model").exists() and str(
        tokenizer_dir
    ) == "tokenizer/smoke":
        resolved_tokenizer_dir = checkpoint_path / "tokenizer"

    tokenizer = load_fast_tokenizer(resolved_tokenizer_dir)
    model = load_model_from_checkpoint(checkpoint_path, device)
    prompts = read_prompts(eval_file)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, prompt in enumerate(prompts, start=1):
            generated = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            record = {
                "index": index,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "checkpoint": str(checkpoint_path),
                "prompt": prompt,
                "generated_text": generated,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    LOGGER.info("Evaluation gespeichert: %s", output_path)
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuehrt eine einfache lokale Smoke-Evaluation aus."
    )
    parser.add_argument(
        "--checkpoint", default="models/smoke/final", help="Lokaler Checkpoint-Ordner."
    )
    parser.add_argument(
        "--tokenizer-dir", default="tokenizer/smoke", help="Lokaler Tokenizer-Ordner."
    )
    parser.add_argument("--eval-file", help="Textdatei mit einem Prompt pro Zeile.")
    parser.add_argument(
        "--eval-dir", default="data/evals", help="Ordner fuer automatische Eval-Dateisuche."
    )
    parser.add_argument(
        "--output-file", default="data/evals/smoke_results.jsonl", help="Zieldatei als JSONL."
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Optionales Zielgeraet.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    eval_file = Path(args.eval_file) if args.eval_file else find_eval_file(args.eval_dir)
    run_evaluation(
        checkpoint=args.checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        eval_file=eval_file,
        output_file=args.output_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )


if __name__ == "__main__":
    main()
