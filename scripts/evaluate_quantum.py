"""Einfache lokale Evaluation fuer quantum-1-base Checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

try:
    from .evaluate import read_prompts
    from .generate import generate_text
    from .generate_quantum import load_quantum_model_from_checkpoint
    from .inspect_model_size import load_quantum_tokenizer_info, load_yaml_config
    from .train_tokenizer import load_fast_tokenizer
    from .train_quantum_pilot import TokenizedTensorDataset
except ImportError:
    from evaluate import read_prompts
    from generate import generate_text
    from generate_quantum import load_quantum_model_from_checkpoint
    from inspect_model_size import load_quantum_tokenizer_info, load_yaml_config
    from train_tokenizer import load_fast_tokenizer
    from train_quantum_pilot import TokenizedTensorDataset


LOGGER = logging.getLogger("lumen.evaluate_quantum")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def default_evaluation_config(config: dict) -> dict:
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    tokenized_dir = Path(data_config.get("tokenized_dir", "data/quantum/final/tokenized"))
    output_dir = Path("data/evals/results/quantum-1-base")
    defaults = {
        "checkpoint_dir": str(Path(training_config.get("output_dir", "models/quantum-1-base")) / "final"),
        "tokenizer_dir": config.get("tokenizer", {}).get("dir", "tokenizer/quantum-1"),
        "validation_file": str(data_config.get("validation_file", tokenized_dir / "validation.pt")),
        "eval_file": "data/evals/quantum_1_base_v1.jsonl",
        "output_dir": str(output_dir),
        "summary_file": "evaluation_summary.json",
        "generations_file": "generations.jsonl",
        "validation_batch_size": int(training_config.get("batch_size", 8)),
        "max_validation_batches": None,
        "generate_samples": True,
        "max_new_tokens": 80,
        "temperature": 0.8,
        "top_p": 0.9,
    }
    defaults.update(config.get("evaluation") or {})
    return defaults


def read_completion_prompts(eval_file: str | Path) -> list[dict]:
    path = Path(eval_file)
    if not path.exists():
        raise FileNotFoundError(f"Eval-Datei nicht gefunden: {path}")
    if path.suffix.lower() == ".jsonl":
        prompts: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Ungueltiges JSONL in {path}:{line_number}") from exc
                prompt = record.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"Eval-Datensatz {path}:{line_number} enthaelt kein nicht-leeres Feld 'prompt'.")
                prompts.append(
                    {
                        "id": record.get("id", f"prompt-{line_number:03d}"),
                        "prompt": prompt.strip(),
                        "category": record.get("category", "completion"),
                    }
                )
        if not prompts:
            raise ValueError(f"Eval-Datei ist leer: {path}")
        return prompts
    return [
        {"id": f"prompt-{index:03d}", "prompt": prompt, "category": "completion"}
        for index, prompt in enumerate(read_prompts(path), start=1)
    ]


@torch.no_grad()
def evaluate_validation_loss(
    model,
    validation_file: str | Path,
    batch_size: int,
    context_length: int,
    max_batches: int | None = None,
) -> dict:
    dataset = TokenizedTensorDataset(validation_file, int(model.config.vocab_size), int(context_length))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    model.eval()
    for batch in loader:
        if max_batches is not None and batches >= int(max_batches):
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        labels = batch["labels"]
        shifted_active_tokens = int((labels[:, 1:] != -100).sum().item())
        if shifted_active_tokens == 0:
            continue
        total_loss += float(outputs.loss.detach().item()) * shifted_active_tokens
        total_tokens += shifted_active_tokens
        batches += 1
    if total_tokens == 0:
        raise ValueError(f"Validation-Datei enthaelt keine auswertbaren Tokens: {validation_file}")
    loss = total_loss / total_tokens
    return {
        "validation_file": str(validation_file),
        "validation_sequences": len(dataset),
        "evaluated_batches": batches,
        "evaluated_tokens": total_tokens,
        "validation_loss": loss,
        "perplexity": math.exp(loss) if loss < 50 else float("inf"),
    }


def run_quantum_evaluation(
    config_path: str | Path,
    checkpoint: str | Path | None = None,
    eval_file: str | Path | None = None,
    output_file: str | Path | None = None,
    device: str | None = None,
) -> Path:
    config = load_yaml_config(config_path)
    evaluation_config = default_evaluation_config(config)
    checkpoint_path = Path(checkpoint or evaluation_config["checkpoint_dir"])
    resolved_eval_file = Path(eval_file or evaluation_config["eval_file"])
    output_dir = Path(evaluation_config["output_dir"])
    summary_file = Path(output_file or output_dir / str(evaluation_config["summary_file"]))
    generations_file = output_dir / str(evaluation_config["generations_file"])
    validation_file = Path(evaluation_config["validation_file"])

    if (checkpoint_path / "tokenizer" / "tokenizer.model").exists():
        tokenizer_dir = checkpoint_path / "tokenizer"
    else:
        tokenizer_dir = Path(evaluation_config.get("tokenizer_dir") or load_quantum_tokenizer_info(config).tokenizer_dir)

    model = load_quantum_model_from_checkpoint(checkpoint_path, device)
    context_length = int(config.get("data", {}).get("block_size") or model.config.max_position_embeddings)
    validation_report = evaluate_validation_loss(
        model=model,
        validation_file=validation_file,
        batch_size=int(evaluation_config["validation_batch_size"]),
        context_length=context_length,
        max_batches=evaluation_config.get("max_validation_batches"),
    )

    generation_count = 0
    if bool(evaluation_config.get("generate_samples", True)):
        tokenizer = load_fast_tokenizer(tokenizer_dir)
        prompts = read_completion_prompts(resolved_eval_file)
        generations_file.parent.mkdir(parents=True, exist_ok=True)
        with generations_file.open("w", encoding="utf-8") as handle:
            for index, prompt_record in enumerate(prompts, start=1):
                generated = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_record["prompt"],
                    max_new_tokens=int(evaluation_config["max_new_tokens"]),
                    temperature=float(evaluation_config["temperature"]),
                    top_p=float(evaluation_config["top_p"]),
                )
                handle.write(
                    json.dumps(
                        {
                            "index": index,
                            "id": prompt_record["id"],
                            "category": prompt_record["category"],
                            "created_at_utc": datetime.now(timezone.utc).isoformat(),
                            "checkpoint": str(checkpoint_path),
                            "prompt": prompt_record["prompt"],
                            "generated_text": generated,
                            "max_new_tokens": int(evaluation_config["max_new_tokens"]),
                            "temperature": float(evaluation_config["temperature"]),
                            "top_p": float(evaluation_config["top_p"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                generation_count += 1

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "tokenizer_dir": str(tokenizer_dir),
        "model_type": "base_language_model",
        "validation": validation_report,
        "generation": {
            "enabled": bool(evaluation_config.get("generate_samples", True)),
            "eval_file": str(resolved_eval_file),
            "generations_file": str(generations_file),
            "num_prompts": generation_count,
        },
    }
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info(
        "quantum Evaluation gespeichert: %s (validation_loss=%.4f, ppl=%.2f)",
        summary_file,
        validation_report["validation_loss"],
        validation_report["perplexity"],
    )
    return summary_file


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
