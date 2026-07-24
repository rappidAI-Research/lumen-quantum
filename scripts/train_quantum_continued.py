"""Weights-only Continued Pretraining fuer quantum-1.6-pilot.

Dieses Skript initialisiert quantum-1.6-pilot AUSSCHLIESSLICH mit den
Modellgewichten aus dem finalen quantum-1-pilot-Modell
(models/quantum-1-base/final). Es werden ausdruecklich NICHT uebernommen:
alter Optimizer-Zustand, alter Scheduler-Zustand, alte globale Schritte.

Es wird ein frischer Optimizer und Scheduler erstellt und bei Schritt 0
begonnen. Ausgabe erfolgt ausschliesslich nach models/quantum-1.6-pilot/.

Standardmaessig startet kein langes Training: --dry-run baut Modell + Init und
macht einen Forward Pass; --max-steps N erlaubt einen kurzen Smoke-Test.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import torch
from accelerate import Accelerator

try:
    import sentencepiece as spm
except ImportError as exc:  # pragma: no cover - depends on local environment.
    raise ImportError(
        "sentencepiece ist erforderlich. Installiere: pip install -r requirements.txt"
    ) from exc

try:
    from .generate_quantum import load_quantum_weights
    from .inspect_model_size import (
        build_quantum_llama_config,
        build_quantum_model,
        count_parameters,
        load_quantum_tokenizer_info,
        load_yaml_config,
        set_reproducible_seed,
    )
    from .quantum_1_6_preflight import EXPECTED_PARAMETER_COUNT, PreflightError, run_preflight
    from .train_quantum_pilot import (
        build_datasets,
        checkpoint_for_step,
        checkpoint_is_complete,
        create_scheduler,
        detect_runtime_environment,
        evaluate_loss,
        log_runtime_environment,
        resolve_resume_checkpoint,
        save_checkpoint,
        save_final_model,
        set_rng_state,
        setup_logging,
    )
except ImportError:
    from generate_quantum import load_quantum_weights
    from inspect_model_size import (
        build_quantum_llama_config,
        build_quantum_model,
        count_parameters,
        load_quantum_tokenizer_info,
        load_yaml_config,
        set_reproducible_seed,
    )
    from quantum_1_6_preflight import EXPECTED_PARAMETER_COUNT, PreflightError, run_preflight
    from train_quantum_pilot import (
        build_datasets,
        checkpoint_for_step,
        checkpoint_is_complete,
        create_scheduler,
        detect_runtime_environment,
        evaluate_loss,
        log_runtime_environment,
        resolve_resume_checkpoint,
        save_checkpoint,
        save_final_model,
        set_rng_state,
        setup_logging,
    )

from torch.utils.data import DataLoader

LOGGER = logging.getLogger("lumen.train_quantum_continued")


def weight_fingerprint(model: torch.nn.Module) -> float:
    """Deterministischer Fingerprint der Eingabe-Embeddings zur Lade-Verifikation."""

    with torch.no_grad():
        return float(model.get_input_embeddings().weight.detach().float().sum().item())


def verify_parameter_count(model: torch.nn.Module) -> int:
    parameter_count = count_parameters(model)
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise PreflightError(
            f"Parameterzahl {parameter_count:,} weicht vom erwarteten Wert {EXPECTED_PARAMETER_COUNT:,} ab. "
            "Architektur muss exakt quantum-1-pilot entsprechen."
        )
    return parameter_count


def initialize_from_base_weights(model: torch.nn.Module, base_model_dir: str | Path) -> dict:
    """Laedt NUR die Modellgewichte aus dem finalen Basismodell (weights-only)."""

    base_dir = Path(base_model_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Basismodell fuer die Initialisierung nicht gefunden: {base_dir}")

    fingerprint_before = weight_fingerprint(model)
    load_quantum_weights(
        model, base_dir
    )  # laedt Gewichte, prueft Architektur, kein from_pretrained
    fingerprint_after = weight_fingerprint(model)

    if fingerprint_before == fingerprint_after:
        raise PreflightError(
            "Weights-only Initialisierung fehlgeschlagen: Gewichte haben sich nach dem Laden nicht veraendert. "
            "Es koennte eine zufaellige Initialisierung statt eines Ladevorgangs vorliegen."
        )
    LOGGER.info("Weights-only Continued Pretraining: Gewichte geladen aus %s", base_dir)
    LOGGER.info("Nicht uebernommen: Optimizer-, Scheduler- und Schrittzustand des alten Laufs.")
    return {
        "base_model_dir": str(base_dir),
        "embedding_sum_before_load": fingerprint_before,
        "embedding_sum_after_load": fingerprint_after,
        "weights_changed": True,
    }


@torch.no_grad()
def log_sample_generations(
    accelerator: Accelerator,
    model,
    tokenizer_model: str | Path,
    prompts: list[str],
    global_step: int,
    log_dir: str | Path,
    sample_file: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
    pad_token_id: int = 3,
) -> None:
    """Generiert feste deutsche Beispiel-Prompts und schreibt sie in ein JSONL-Log.

    Bewusst robust: Fehler beim Generieren duerfen das Training nicht abbrechen.
    """

    if not prompts or not accelerator.is_main_process:
        return
    try:
        sp = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
        unwrapped = accelerator.unwrap_model(model)
        was_training = unwrapped.training
        unwrapped.eval()
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        records = []
        for prompt in prompts:
            ids = [bos_token_id, *sp.encode(prompt, out_type=int)]
            input_ids = torch.tensor([ids], dtype=torch.long, device=accelerator.device)
            output = unwrapped.generate(
                input_ids=input_ids,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            new_tokens = output[0].tolist()[len(ids) :]
            completion = sp.decode(new_tokens)
            records.append(
                {
                    "global_step": global_step,
                    "created_at_utc": datetime.now(UTC).isoformat(),
                    "prompt": prompt,
                    "completion": completion,
                }
            )
        with (log_path / sample_file).open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if was_training:
            unwrapped.train()
        LOGGER.info(
            "Beispiel-Generierungen fuer Schritt %d geschrieben (%d Prompts).",
            global_step,
            len(records),
        )
    except Exception as error:  # pragma: no cover - Generierung ist optionales Logging.
        LOGGER.warning("Beispiel-Generierung bei Schritt %d fehlgeschlagen: %s", global_step, error)


def assert_output_dir(output_dir: Path) -> None:
    normalized = output_dir.as_posix().rstrip("/")
    if not normalized.startswith("models/quantum-1.6-pilot"):
        raise PreflightError(
            f"Output darf ausschliesslich nach models/quantum-1.6-pilot/ geschrieben werden, nicht: {normalized}."
        )


def train(
    config_path: str | Path,
    data_config_path: str | Path | None = None,
    resume_from: str | None = None,
    max_steps_override: int | None = None,
    dry_run: bool = False,
) -> Path:
    config = load_yaml_config(config_path)
    training_config = config["training"]
    output_dir = Path(training_config["output_dir"])
    assert_output_dir(output_dir)
    setup_logging(None if dry_run else output_dir)
    set_reproducible_seed(int(config["seed"]))

    # 1) Torch-freie Preflight-Checks (Config, Pfade, Tokenizer-Hash, Basismodell).
    preflight = run_preflight(config_path, data_config_path)
    LOGGER.info(
        "Preflight OK. Tokenizer identisch zum Basismodell: %s",
        preflight["tokenizer"]["identical_to_base_model"],
    )
    for note in preflight.get("notes", []):
        LOGGER.info(note)

    runtime = detect_runtime_environment(str(training_config.get("mixed_precision", "auto")))
    log_runtime_environment(runtime)

    tokenizer_info = load_quantum_tokenizer_info(config)
    llama_config = build_quantum_llama_config(config, tokenizer_info)
    model = build_quantum_model(llama_config)

    # 2) Parameterzahl exakt pruefen (keine zufaellige Neuinitialisierung uebernehmen).
    parameter_count = verify_parameter_count(model)
    LOGGER.info("Parameterzahl exakt geprueft: %s", f"{parameter_count:,}")

    # 3) Resume eines eigenen 1.6-Laufs? Sonst weights-only Init aus dem Basismodell.
    resume_value = (
        resume_from if resume_from is not None else training_config.get("resume_from_checkpoint")
    )
    resume_checkpoint = resolve_resume_checkpoint(output_dir, resume_value)
    init_report: dict = {}
    if resume_checkpoint is None:
        init_report = initialize_from_base_weights(model, config["init"]["from_model"])
    else:
        LOGGER.info("Setze eigenen quantum-1.6-pilot-Lauf fort: %s", resume_checkpoint)
        load_quantum_weights(model, resume_checkpoint)

    # 4) Dry-Run: Forward Pass ueber einen synthetischen Batch, nichts speichern.
    if dry_run:
        device = torch.device(runtime["device"])
        model.to(device)
        model.eval()
        seq_len = int(config["data"]["block_size"])
        input_ids = torch.randint(
            0, tokenizer_info.vocab_size, (1, seq_len), dtype=torch.long, device=device
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        if outputs.loss is None or not torch.isfinite(outputs.loss):
            raise ValueError("Dry-Run Forward Pass lieferte keinen endlichen Loss.")
        LOGGER.info(
            "Dry-Run erfolgreich: loss=%.4f, Init aus %s, keine Gewichte gespeichert.",
            float(outputs.loss.item()),
            init_report.get("base_model_dir", resume_checkpoint),
        )
        return output_dir

    max_steps = int(
        max_steps_override if max_steps_override is not None else training_config["max_steps"]
    )
    if max_steps <= 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "max_steps=%d: Kein Training gestartet. Nutze --max-steps 1 oder --dry-run.", max_steps
        )
        return output_dir

    train_dataset, validation_dataset, data_stats = build_datasets(
        config, tokenizer_info, llama_config
    )
    LOGGER.info(
        "Train-Sequenzen: %d | Validation-Sequenzen: %d",
        len(train_dataset),
        len(validation_dataset),
    )
    if "test" in data_stats:
        LOGGER.info("Test-Sequenzen geprueft: %d", data_stats["test"]["sequences"])

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

    # 5) Frischer Optimizer und Scheduler (kein alter Zustand).
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = create_scheduler(optimizer, training_config, max_steps)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        mixed_precision=str(runtime["mixed_precision"]),
    )

    start_step = 0
    start_epoch = 0
    if resume_checkpoint is not None:
        # Nur beim Fortsetzen EINES EIGENEN 1.6-Laufs wird der Trainingszustand geladen.
        state = torch.load(
            resume_checkpoint / "training_state.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["global_step"])
        start_epoch = int(state.get("epoch", 0))
        set_rng_state(state.get("rng_state"))
        LOGGER.info("Trainingszustand fortgesetzt bei Schritt %d.", start_step)

    model, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, validation_loader, scheduler
    )
    model.train()

    log_config = config.get("logging", {})
    eval_config = config.get("evaluation", {})
    sample_prompts = list(eval_config.get("sample_prompts", []))
    global_step = start_step
    epoch = start_epoch
    running_loss = 0.0
    LOGGER.info(
        "Starte Continued Pretraining bis Schritt %d auf %s.", max_steps, accelerator.device
    )

    while global_step < max_steps:
        epoch += 1
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), float(training_config["max_grad_norm"])
                    )
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
                    LOGGER.info(
                        "step=%d validation_loss=%.4f",
                        global_step,
                        evaluate_loss(model, validation_loader, accelerator),
                    )
                    log_sample_generations(
                        accelerator,
                        model,
                        tokenizer_info.tokenizer_model,
                        sample_prompts,
                        global_step,
                        log_config.get("log_dir", "logs/quantum_1_6_pilot"),
                        str(log_config.get("sample_generations_file", "sample_generations.jsonl")),
                        int(eval_config.get("sample_max_new_tokens", 64)),
                        float(eval_config.get("temperature", 0.8)),
                        float(eval_config.get("top_p", 0.9)),
                        bos_token_id=tokenizer_info.bos_token_id,
                        eos_token_id=tokenizer_info.eos_token_id,
                        pad_token_id=tokenizer_info.pad_token_id,
                    )
                if (
                    accelerator.is_main_process
                    and global_step % int(training_config["save_steps"]) == 0
                ):
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
    return save_final_model(
        accelerator, model, output_dir, config, tokenizer_info.tokenizer_dir, global_step
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weights-only Continued Pretraining fuer quantum-1.6-pilot."
    )
    parser.add_argument("--config", default="configs/quantum_1_6_pilot_train.yaml")
    parser.add_argument("--data-config", default="configs/quantum_1_6_pilot_data.yaml")
    parser.add_argument(
        "--resume-from", help="Checkpoint-Ordner eines eigenen 1.6-Laufs oder 'auto'."
    )
    parser.add_argument(
        "--max-steps", type=int, help="Maximale Trainingsschritte fuer einen Smoke-Test."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Init + Forward Pass, nichts speichern."
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    train(
        args.config,
        data_config_path=args.data_config,
        resume_from=args.resume_from,
        max_steps_override=args.max_steps,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
