from pathlib import Path

import torch
from accelerate import Accelerator
from transformers import LlamaConfig, LlamaForCausalLM

from scripts.train_smoke import (
    checkpoint_for_step,
    checkpoint_is_complete,
    latest_checkpoint,
    resolve_resume_checkpoint,
    save_checkpoint,
    save_final_model,
)


def _tiny_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=16,
        pad_token_id=2,
        bos_token_id=0,
        eos_token_id=1,
    )
    return LlamaForCausalLM(config)


def _run_one_optimizer_step(model, optimizer, scheduler) -> None:
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8), dtype=torch.long)
    outputs = model(input_ids=input_ids, labels=input_ids)
    outputs.loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def _save_tiny_checkpoint(tmp_path: Path, global_step: int) -> Path:
    output_dir = tmp_path / "models" / "smoke"
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer.model").write_bytes(b"fake-sentencepiece-model")
    (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tokenizer_dir / "special_tokens_map.json").write_text("{}", encoding="utf-8")

    accelerator = Accelerator(cpu=True)
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    _run_one_optimizer_step(model, optimizer, scheduler)

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        global_step=global_step,
        epoch=1,
        config={"seed": 123, "training": {"output_dir": str(output_dir)}},
        tokenizer_dir=tokenizer_dir,
    )
    return output_dir


def test_resume_checkpoint_exists_after_short_training_step(tmp_path):
    output_dir = _save_tiny_checkpoint(tmp_path, global_step=5)
    checkpoint = checkpoint_for_step(output_dir, 5)
    temp_checkpoint = checkpoint.with_name(f"{checkpoint.name}.tmp")

    assert checkpoint.name == "checkpoint-step-00005"
    assert checkpoint_is_complete(checkpoint)
    assert not temp_checkpoint.exists()
    assert (checkpoint / "training_state.pt").exists()
    assert (checkpoint / "config.json").exists()
    assert (checkpoint / "model.safetensors").exists()
    assert (checkpoint / "tokenizer.model").exists()
    assert (checkpoint / "tokenizer" / "tokenizer.model").exists()

    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["global_step"] == 5
    assert state["epoch"] == 1
    assert "optimizer" in state
    assert "scheduler" in state
    assert "rng_state" in state
    assert "config" in state


def test_auto_resume_finds_latest_complete_checkpoint(tmp_path):
    output_dir = _save_tiny_checkpoint(tmp_path, global_step=5)
    _save_tiny_checkpoint(tmp_path, global_step=12)

    assert latest_checkpoint(output_dir) == checkpoint_for_step(output_dir, 12)
    assert resolve_resume_checkpoint(output_dir, "auto") == checkpoint_for_step(output_dir, 12)


def test_final_model_contains_sentencepiece_tokenizer_in_root(tmp_path):
    output_dir = tmp_path / "models" / "smoke"
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "tokenizer.model").write_bytes(b"fake-sentencepiece-model")
    (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tokenizer_dir / "special_tokens_map.json").write_text("{}", encoding="utf-8")

    accelerator = Accelerator(cpu=True)
    model = _tiny_model()
    final_dir = save_final_model(
        accelerator=accelerator,
        model=model,
        output_dir=output_dir,
        config={"seed": 123},
        tokenizer_dir=tokenizer_dir,
        global_step=1,
    )

    assert (final_dir / "config.json").exists()
    assert (final_dir / "model.safetensors").exists()
    assert (final_dir / "tokenizer.model").exists()
    assert (final_dir / "tokenizer" / "tokenizer.model").exists()
