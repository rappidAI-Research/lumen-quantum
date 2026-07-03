"""Bereitet Pilot-Training fuer quantum-1-base mit vollstaendigen Checkpoints vor.

Standardmaessig ist max_steps in der Config auf 0 gesetzt, damit kein langes
Training versehentlich startet. Fuer einen lokalen Funktionstest kann
--max-steps 1 genutzt werden.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

try:
    import sentencepiece as spm
except ImportError as exc:  # pragma: no cover - depends on local environment.
    raise ImportError("sentencepiece ist erforderlich. Installiere: pip install -r requirements.txt") from exc

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
    from .train_smoke import finalize_checkpoint_dir, remove_path_with_retries
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
    from train_smoke import finalize_checkpoint_dir, remove_path_with_retries


LOGGER = logging.getLogger("lumen.train_quantum_pilot")
CHECKPOINT_PREFIX = "checkpoint-step-"


class JsonlTokenBlockDataset(Dataset):
    """Kleine In-Memory-Dataset-Klasse fuer Pilotlaeufe."""

    def __init__(self, path: str | Path, tokenizer_model: str | Path, text_field: str, block_size: int, pad_token_id: int):
        data_path = Path(path)
        if not data_path.exists():
            raise FileNotFoundError(f"JSONL-Datendatei nicht gefunden: {data_path}")

        sp = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
        self.examples: list[dict[str, torch.Tensor]] = []
        for text in iter_jsonl_texts(data_path, text_field):
            ids = [1, *sp.encode(text, out_type=int), 2]
            for start in range(0, len(ids), block_size):
                chunk = ids[start : start + block_size]
                if len(chunk) < 2:
                    continue
                attention_mask = [1] * len(chunk)
                if len(chunk) < block_size:
                    pad_length = block_size - len(chunk)
                    chunk = [*chunk, *([pad_token_id] * pad_length)]
                    attention_mask = [*attention_mask, *([0] * pad_length)]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                mask = torch.tensor(attention_mask, dtype=torch.long)
                labels = input_ids.clone()
                labels[mask == 0] = -100
                self.examples.append({"input_ids": input_ids, "attention_mask": mask, "labels": labels})

        if not self.examples:
            raise ValueError(f"Keine Trainingsbloecke aus {data_path} erzeugt.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def iter_jsonl_texts(path: Path, text_field: str) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ungueltiges JSONL in {path}:{line_number}") from exc
            text = record.get(text_field)
            if isinstance(text, str) and text.strip():
                yield text.strip()


def setup_logging(output_dir: str | Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(output / "train.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def get_rng_state() -> dict:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def create_scheduler(optimizer, training_config: dict, total_steps: int):
    warmup_steps = min(int(training_config["warmup_steps"]), max(0, total_steps - 1))
    scheduler_name = str(training_config.get("lr_scheduler", "cosine")).lower()
    if scheduler_name == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if scheduler_name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    raise ValueError(f"Unbekannter lr_scheduler: {scheduler_name}. Erlaubt: cosine, linear.")


def checkpoint_for_step(output_dir: str | Path, global_step: int) -> Path:
    return Path(output_dir) / "checkpoints" / f"{CHECKPOINT_PREFIX}{global_step:05d}"


def checkpoint_step(checkpoint_dir: str | Path) -> int:
    name = Path(checkpoint_dir).name
    if not name.startswith(CHECKPOINT_PREFIX):
        return -1
    step_text = name.removeprefix(CHECKPOINT_PREFIX)
    return int(step_text) if step_text.isdigit() else -1


def checkpoint_is_complete(checkpoint_dir: str | Path) -> bool:
    checkpoint = Path(checkpoint_dir)
    has_weights = (
        (checkpoint / "model.safetensors").exists()
        or (checkpoint / "pytorch_model.bin").exists()
        or (checkpoint / "model.safetensors.index.json").exists()
        or (checkpoint / "pytorch_model.bin.index.json").exists()
    )
    return (
        checkpoint.is_dir()
        and has_weights
        and (checkpoint / "config.json").exists()
        and (checkpoint / "training_state.pt").exists()
        and (checkpoint / "tokenizer_manifest.json").exists()
    )


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    checkpoint_root = Path(output_dir) / "checkpoints"
    if not checkpoint_root.exists():
        return None
    checkpoints = sorted(
        (path for path in checkpoint_root.glob(f"{CHECKPOINT_PREFIX}*") if checkpoint_is_complete(path)),
        key=checkpoint_step,
    )
    return checkpoints[-1] if checkpoints else None


def resolve_resume_checkpoint(output_dir: str | Path, resume_value: str | None) -> Path | None:
    if not resume_value:
        return None
    if resume_value.lower() == "auto":
        checkpoint = latest_checkpoint(output_dir)
        if checkpoint is None:
            raise FileNotFoundError(
                f"Kein vollstaendiger Checkpoint zum Fortsetzen in {Path(output_dir) / 'checkpoints'} gefunden."
            )
        return checkpoint
    checkpoint = Path(resume_value)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Resume-Checkpoint nicht gefunden: {checkpoint}")
    checkpoint_dir = checkpoint if checkpoint.is_dir() else checkpoint.parent
    if not checkpoint_is_complete(checkpoint_dir):
        raise FileNotFoundError(f"Resume-Checkpoint ist unvollstaendig: {checkpoint_dir}")
    return checkpoint_dir


def copy_quantum_tokenizer(source_dir: str | Path, target_dir: str | Path) -> None:
    source = Path(source_dir)
    target = Path(target_dir)
    if not source.exists():
        raise FileNotFoundError(f"Tokenizer-Ordner nicht gefunden: {source}")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def copy_quantum_tokenizer_files_to_root(source_dir: str | Path, model_dir: str | Path) -> None:
    source = Path(source_dir)
    target = Path(model_dir)
    for filename in [
        "tokenizer.model",
        "tokenizer.vocab",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer_manifest.json",
        "validation_report.json",
    ]:
        source_file = source / filename
        if source_file.exists():
            shutil.copy2(source_file, target / filename)


def save_checkpoint(
    accelerator: Accelerator,
    model,
    optimizer,
    scheduler,
    output_dir: str | Path,
    global_step: int,
    epoch: int,
    config: dict,
    tokenizer_dir: str | Path,
) -> Path:
    checkpoint_dir = checkpoint_for_step(output_dir, global_step)
    temp_dir = checkpoint_dir.with_name(f"{checkpoint_dir.name}.tmp")
    remove_path_with_retries(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        str(temp_dir),
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
        safe_serialization=True,
    )
    tokenizer_manifest = json.loads((Path(tokenizer_dir) / "tokenizer_manifest.json").read_text(encoding="utf-8"))
    training_state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "rng_state": get_rng_state(),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "tokenizer_manifest": tokenizer_manifest,
    }
    accelerator.save(training_state, temp_dir / "training_state.pt")
    copy_quantum_tokenizer(tokenizer_dir, temp_dir / "tokenizer")
    copy_quantum_tokenizer_files_to_root(tokenizer_dir, temp_dir)
    finalize_checkpoint_dir(temp_dir, checkpoint_dir)
    (Path(output_dir) / "latest_checkpoint.txt").write_text(str(checkpoint_dir), encoding="utf-8")
    LOGGER.info("Checkpoint gespeichert: %s", checkpoint_dir)
    return checkpoint_dir


def save_final_model(
    accelerator: Accelerator,
    model,
    output_dir: str | Path,
    config: dict,
    tokenizer_dir: str | Path,
    global_step: int,
) -> Path:
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        str(final_dir),
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
        safe_serialization=True,
    )
    copy_quantum_tokenizer(tokenizer_dir, final_dir / "tokenizer")
    copy_quantum_tokenizer_files_to_root(tokenizer_dir, final_dir)
    metadata = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "global_step": global_step,
        "note": "Lokales quantum-1-base Pilot-Artefakt; keine vortrainierten Gewichte wurden geladen.",
        "config": config,
    }
    (final_dir / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Finales Modell gespeichert: %s", final_dir)
    return final_dir


@torch.no_grad()
def evaluate_loss(model, dataloader: DataLoader, accelerator: Accelerator) -> float:
    model.eval()
    losses: list[torch.Tensor] = []
    for batch in dataloader:
        outputs = model(**batch)
        gathered = accelerator.gather_for_metrics(outputs.loss.detach().repeat(batch["input_ids"].shape[0]))
        losses.append(gathered)
    model.train()
    if not losses:
        return math.nan
    return float(torch.cat(losses).mean().item())


def train(config_path: str | Path, resume_from: str | None = None, max_steps_override: int | None = None) -> Path:
    config = load_yaml_config(config_path)
    training_config = config["training"]
    output_dir = Path(training_config["output_dir"])
    setup_logging(output_dir)
    set_reproducible_seed(int(config["seed"]))

    size_report = inspect_model_size(config_path)
    LOGGER.info("Parameterzahl geprueft: %s", f"{size_report['parameter_count']:,}")
    tokenizer_info = load_quantum_tokenizer_info(config)
    llama_config = build_quantum_llama_config(config, tokenizer_info)
    model = build_quantum_model(llama_config)
    LOGGER.info("Modell initialisiert mit zufaelligen Gewichten: %s Parameter.", f"{count_parameters(model):,}")

    max_steps = int(max_steps_override if max_steps_override is not None else training_config["max_steps"])
    if max_steps <= 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("max_steps=%d: Kein Training gestartet. Nutze --max-steps 1 fuer einen lokalen Pilotlauf.", max_steps)
        return output_dir

    data_config = config["data"]
    block_size = int(data_config["block_size"])
    if block_size > int(llama_config.max_position_embeddings):
        raise ValueError("data.block_size ist groesser als max_position_embeddings.")

    train_dataset = JsonlTokenBlockDataset(
        data_config["train_file"],
        tokenizer_info.tokenizer_model,
        str(data_config.get("text_field", "text")),
        block_size,
        tokenizer_info.pad_token_id,
    )
    validation_dataset = JsonlTokenBlockDataset(
        data_config["validation_file"],
        tokenizer_info.tokenizer_model,
        str(data_config.get("text_field", "text")),
        block_size,
        tokenizer_info.pad_token_id,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config["num_workers"]),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=int(training_config["num_workers"]),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = create_scheduler(optimizer, training_config, max_steps)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        mixed_precision=str(training_config.get("mixed_precision", "no")),
    )

    resume_value = resume_from if resume_from is not None else training_config.get("resume_from_checkpoint")
    checkpoint_path = resolve_resume_checkpoint(output_dir, resume_value)
    start_step = 0
    start_epoch = 0
    if checkpoint_path is not None:
        checkpoint_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
        LOGGER.info("Lade lokalen Resume-Checkpoint: %s", checkpoint_dir)
        load_quantum_weights(model, checkpoint_dir)
        state = torch.load(checkpoint_dir / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["global_step"])
        start_epoch = int(state.get("epoch", 0))
        set_rng_state(state.get("rng_state"))
        LOGGER.info("Setze Training bei Schritt %d fort (Epoch %d).", start_step, start_epoch)

    model, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, validation_loader, scheduler
    )
    model.train()

    global_step = start_step
    epoch = start_epoch
    running_loss = 0.0
    LOGGER.info("Starte Pilot-Training bis Schritt %d auf %s.", max_steps, accelerator.device)
    while global_step < max_steps:
        epoch += 1
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(training_config["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                running_loss += float(loss.detach().item())
                if global_step % int(training_config["logging_steps"]) == 0:
                    LOGGER.info(
                        "step=%d loss=%.4f lr=%.8f epoch=%d",
                        global_step,
                        running_loss / int(training_config["logging_steps"]),
                        scheduler.get_last_lr()[0],
                        epoch,
                    )
                    running_loss = 0.0
                if global_step % int(training_config["eval_steps"]) == 0:
                    LOGGER.info("step=%d validation_loss=%.4f", global_step, evaluate_loss(model, validation_loader, accelerator))
                if accelerator.is_main_process and global_step % int(training_config["save_steps"]) == 0:
                    save_checkpoint(
                        accelerator,
                        model,
                        optimizer,
                        scheduler,
                        output_dir,
                        global_step,
                        epoch,
                        config,
                        tokenizer_info.tokenizer_dir,
                    )
                if global_step >= max_steps:
                    break

    if accelerator.is_main_process:
        final_checkpoint = checkpoint_for_step(output_dir, global_step)
        if not checkpoint_is_complete(final_checkpoint):
            save_checkpoint(
                accelerator,
                model,
                optimizer,
                scheduler,
                output_dir,
                global_step,
                epoch,
                config,
                tokenizer_info.tokenizer_dir,
            )

    accelerator.wait_for_everyone()
    return save_final_model(accelerator, model, output_dir, config, tokenizer_info.tokenizer_dir, global_step)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pilot-Training fuer quantum-1-base vorbereiten/kurz testen.")
    parser.add_argument("--config", default="configs/quantum_1_base_pilot.yaml")
    parser.add_argument("--resume-from", help="Checkpoint-Ordner, training_state.pt oder 'auto'.")
    parser.add_argument("--max-steps", type=int, help="Maximale Trainingsschritte fuer einen lokalen Pilotlauf.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    train(args.config, resume_from=args.resume_from, max_steps_override=args.max_steps)


if __name__ == "__main__":
    main()
