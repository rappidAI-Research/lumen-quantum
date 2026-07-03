"""Trainiert das Lumen-Quantum-Smoke-Modell mit zufaelligen Gewichten.

Wichtig: Dieses Skript verwendet absichtlich niemals from_pretrained fuer ein
Modell. LlamaForCausalLM wird direkt aus LlamaConfig erzeugt und startet damit
mit zufaellig initialisierten Gewichten.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

try:
    from .train_tokenizer import load_fast_tokenizer
    from .generate import load_model_state_dict
except ImportError:
    from train_tokenizer import load_fast_tokenizer
    from generate import load_model_state_dict


LOGGER = logging.getLogger("lumen.train_smoke")
CHECKPOINT_PREFIX = "checkpoint-step-"


class TokenBlockDataset(Dataset):
    def __init__(self, path: str | Path, pad_token_id: int):
        data_path = Path(path)
        if not data_path.exists():
            raise FileNotFoundError(f"Tokenisierte Daten nicht gefunden: {data_path}")
        data = torch.load(data_path, map_location="cpu", weights_only=True)
        self.input_ids = data["input_ids"].long()
        self.attention_mask = data["attention_mask"].long()
        self.pad_token_id = int(pad_token_id)
        if self.input_ids.ndim != 2:
            raise ValueError(f"input_ids in {data_path} muss zweidimensional sein.")
        if self.input_ids.shape != self.attention_mask.shape:
            raise ValueError("input_ids und attention_mask haben unterschiedliche Formen.")

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids = self.input_ids[index]
        attention_mask = self.attention_mask[index]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


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


def load_yaml_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state() -> dict:
    """Sammelt den RNG-Zustand, damit ein Resume die Datenreihenfolge fortsetzt."""

    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
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


def build_llama_config(
    config: dict,
    vocab_size: int | None = None,
    pad_token_id: int | None = None,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> LlamaConfig:
    model_config = config["model"]
    data_config = config.get("data", {})
    resolved_vocab_size = int(vocab_size or model_config["vocab_size"])
    max_positions = int(model_config.get("max_position_embeddings", data_config.get("block_size", 256)))

    return LlamaConfig(
        vocab_size=resolved_vocab_size,
        hidden_size=int(model_config["hidden_size"]),
        intermediate_size=int(model_config["intermediate_size"]),
        num_hidden_layers=int(model_config["num_hidden_layers"]),
        num_attention_heads=int(model_config["num_attention_heads"]),
        num_key_value_heads=int(model_config["num_key_value_heads"]),
        max_position_embeddings=max_positions,
        rms_norm_eps=float(model_config["rms_norm_eps"]),
        rope_theta=float(model_config["rope_theta"]),
        tie_word_embeddings=bool(model_config.get("tie_word_embeddings", False)),
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )


def build_model(llama_config: LlamaConfig) -> LlamaForCausalLM:
    # Direkte Konstruktion aus Config: keine externen oder vortrainierten Gewichte.
    return LlamaForCausalLM(llama_config)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def load_tokenized_metadata(data_dir: str | Path) -> dict:
    metadata_path = Path(data_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadaten nicht gefunden: {metadata_path}. Fuehre zuerst scripts/prepare_data.py aus."
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    state_file = checkpoint / "training_state.pt"
    config_file = checkpoint / "config.json"
    has_weights = (
        (checkpoint / "model.safetensors").exists()
        or (checkpoint / "pytorch_model.bin").exists()
        or (checkpoint / "model.safetensors.index.json").exists()
        or (checkpoint / "pytorch_model.bin.index.json").exists()
    )
    return checkpoint.is_dir() and state_file.exists() and config_file.exists() and has_weights


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
                f"Kein vollstaendiger Checkpoint zum Fortsetzen in "
                f"{Path(output_dir) / 'checkpoints'} gefunden."
            )
        return checkpoint
    checkpoint = Path(resume_value)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Resume-Checkpoint nicht gefunden: {checkpoint}")
    if checkpoint.is_dir() and not checkpoint_is_complete(checkpoint):
        raise FileNotFoundError(f"Resume-Checkpoint ist unvollstaendig: {checkpoint}")
    return checkpoint


def remove_path_with_retries(path: str | Path, attempts: int = 5, delay_seconds: float = 0.2) -> None:
    target = Path(path)
    if not target.exists():
        return

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            return
        except PermissionError as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(delay_seconds)

    raise PermissionError(f"Konnte {target} nach {attempts} Versuchen nicht entfernen: {last_error}")


def finalize_checkpoint_dir(temp_dir: str | Path, checkpoint_dir: str | Path) -> None:
    temp = Path(temp_dir)
    target = Path(checkpoint_dir)
    if not temp.exists():
        raise FileNotFoundError(f"Temporaerer Checkpoint-Ordner fehlt: {temp}")

    remove_path_with_retries(target)
    try:
        shutil.move(str(temp), str(target))
    except PermissionError as exc:
        raise PermissionError(
            "Checkpoint konnte unter Windows nicht finalisiert werden. "
            f"Temp-Ordner: {temp}; Ziel: {target}. "
            "Falls ein Virenscanner oder Explorer-Fenster den Ordner blockiert, "
            "schliesse es kurz und starte den Trainingsbefehl erneut."
        ) from exc


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
) -> None:
    checkpoint_dir = checkpoint_for_step(output_dir, global_step)
    temp_dir = checkpoint_dir.with_name(f"{checkpoint_dir.name}.tmp")
    remove_path_with_retries(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    # Gewichte + config.json (inkl. architectures) + generation_config.json als
    # safetensors ablegen -> direkt HF-ladbar und GGUF-konvertierbar.
    unwrapped.save_pretrained(
        str(temp_dir),
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
        safe_serialization=True,
    )
    # Trainingszustand separat: Optimizer, Scheduler, Schrittzaehler und RNG-State.
    training_state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "rng_state": get_rng_state(),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
    }
    accelerator.save(training_state, temp_dir / "training_state.pt")
    copy_tokenizer(tokenizer_dir, temp_dir / "tokenizer")
    copy_tokenizer_files_to_model_root(tokenizer_dir, temp_dir)
    finalize_checkpoint_dir(temp_dir, checkpoint_dir)
    latest_file = Path(output_dir) / "latest_checkpoint.txt"
    latest_file.write_text(str(checkpoint_dir), encoding="utf-8")
    LOGGER.info("Checkpoint gespeichert: %s", checkpoint_dir)


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
    # Vollstaendiges, HF-/GGUF-kompatibles Modellverzeichnis:
    # config.json (inkl. architectures), model.safetensors, generation_config.json.
    unwrapped.save_pretrained(
        str(final_dir),
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
        safe_serialization=True,
    )
    copy_tokenizer(tokenizer_dir, final_dir / "tokenizer")
    copy_tokenizer_files_to_model_root(tokenizer_dir, final_dir)
    metadata = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "global_step": global_step,
        "note": "Lokaler Smoke-Checkpoint; keine vortrainierten Gewichte wurden geladen.",
        "config": config,
    }
    with (final_dir / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Finales Modell gespeichert: %s", final_dir)
    return final_dir


def copy_tokenizer(source_dir: str | Path, target_dir: str | Path) -> None:
    source = Path(source_dir)
    target = Path(target_dir)
    if not source.exists():
        LOGGER.warning("Tokenizer-Ordner fehlt und wird nicht kopiert: %s", source)
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def copy_tokenizer_files_to_model_root(source_dir: str | Path, model_dir: str | Path) -> None:
    """Kopiert llama.cpp-relevante Tokenizer-Dateien direkt ins Modellverzeichnis."""

    source = Path(source_dir)
    target = Path(model_dir)
    files = [
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ]
    for filename in files:
        source_file = source / filename
        if source_file.exists():
            shutil.copy2(source_file, target / filename)


@torch.no_grad()
def evaluate_loss(model, dataloader: DataLoader, accelerator: Accelerator) -> float:
    model.eval()
    losses: list[torch.Tensor] = []
    for batch in dataloader:
        outputs = model(**batch)
        loss = outputs.loss.detach()
        gathered = accelerator.gather_for_metrics(loss.repeat(batch["input_ids"].shape[0]))
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

    tokenizer = load_fast_tokenizer(training_config["tokenizer_dir"], config["tokenizer"].get("special_tokens"))
    tokenized_metadata = load_tokenized_metadata(training_config["data_dir"])
    block_size = int(tokenized_metadata["block_size"])
    if block_size > int(config["model"]["max_position_embeddings"]):
        raise ValueError("block_size ist groesser als max_position_embeddings in der Modell-Config.")

    llama_config = build_llama_config(
        config,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = build_model(llama_config)
    parameter_count = count_parameters(model)
    LOGGER.info("Modell initialisiert mit zufaelligen Gewichten: %s Parameter.", f"{parameter_count:,}")

    train_dataset = TokenBlockDataset(Path(training_config["data_dir"]) / "train.pt", tokenizer.pad_token_id)
    validation_dataset = TokenBlockDataset(Path(training_config["data_dir"]) / "validation.pt", tokenizer.pad_token_id)
    if len(train_dataset) == 0:
        raise ValueError("Trainingsdataset ist leer.")

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

    max_steps = int(max_steps_override or training_config["max_steps"])
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
        state_path = checkpoint_dir / "training_state.pt"
        LOGGER.info("Lade lokalen Resume-Checkpoint: %s", checkpoint_dir)
        # Gewichte aus dem lokalen safetensors-Checkpoint (kein from_pretrained).
        model.load_state_dict(load_model_state_dict(checkpoint_dir))
        # Eigener, vertrauenswuerdiger Checkpoint -> weights_only=False noetig fuer
        # Optimizer-/Scheduler-/RNG-Objekte (Python-Objekte, keine reinen Tensoren).
        state = torch.load(state_path, map_location="cpu", weights_only=False)
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

    logging_steps = int(training_config["logging_steps"])
    eval_steps = int(training_config["eval_steps"])
    save_steps = int(training_config["save_steps"])
    max_grad_norm = float(training_config["max_grad_norm"])
    global_step = start_step
    running_loss = 0.0
    epoch = start_epoch

    LOGGER.info("Starte Training bis Schritt %d auf %s.", max_steps, accelerator.device)
    while global_step < max_steps:
        epoch += 1
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                running_loss += float(loss.detach().item())
                if global_step % logging_steps == 0:
                    average_loss = running_loss / logging_steps
                    running_loss = 0.0
                    current_lr = scheduler.get_last_lr()[0]
                    LOGGER.info(
                        "step=%d loss=%.4f lr=%.8f epoch=%d",
                        global_step,
                        average_loss,
                        current_lr,
                        epoch,
                    )

                if global_step % eval_steps == 0:
                    validation_loss = evaluate_loss(model, validation_loader, accelerator)
                    LOGGER.info("step=%d validation_loss=%.4f", global_step, validation_loss)

                if accelerator.is_main_process and global_step % save_steps == 0:
                    save_checkpoint(
                        accelerator,
                        model,
                        optimizer,
                        scheduler,
                        output_dir,
                        global_step,
                        epoch,
                        config,
                        training_config["tokenizer_dir"],
                    )

                if global_step >= max_steps:
                    break

    if accelerator.is_main_process and global_step > 0:
        final_resume_checkpoint = checkpoint_for_step(output_dir, global_step)
        if not checkpoint_is_complete(final_resume_checkpoint):
            save_checkpoint(
                accelerator,
                model,
                optimizer,
                scheduler,
                output_dir,
                global_step,
                epoch,
                config,
                training_config["tokenizer_dir"],
            )

    accelerator.wait_for_everyone()
    final_dir = save_final_model(
        accelerator,
        model,
        output_dir,
        config,
        training_config["tokenizer_dir"],
        global_step,
    )
    return final_dir


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trainiert das Lumen-Smoke-Modell.")
    parser.add_argument("--config", default="configs/smoke_5m.yaml", help="Pfad zur YAML-Konfiguration.")
    parser.add_argument(
        "--resume-from",
        help="Checkpoint-Ordner, training_state.pt oder 'auto'. Ueberschreibt training.resume_from_checkpoint.",
    )
    parser.add_argument("--max-steps", type=int, help="Maximale Trainingsschritte fuer schnelle Tests.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    train(args.config, resume_from=args.resume_from, max_steps_override=args.max_steps)


if __name__ == "__main__":
    main()
