"""Bounded resumable Base-training smoke for Quantum 1 Echelon.

This exercises the production-facing contracts on tiny or explicitly supplied
configs. It is not the 40B production trainer and refuses unbounded execution.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import LlamaForCausalLM

from scripts.echelon_checkpoint_manifest import build_manifest, write_atomic
from scripts.echelon_preflight import build_hf_config
from scripts.echelon_shard_stream import ShardedTokenStream


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config root must be a mapping")
    return payload


def save_checkpoint(
    directory: Path,
    *,
    model: LlamaForCausalLM,
    optimizer: AdamW,
    scheduler: LambdaLR,
    global_step: int,
    processed_tokens: int,
    data_offset: int,
    run_id: str,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(directory / "model.safetensors"))
    torch.save(optimizer.state_dict(), directory / "optimizer.pt")
    torch.save(scheduler.state_dict(), directory / "scheduler.pt")
    torch.save(torch.get_rng_state(), directory / "torch_rng.pt")
    state = {
        "schema_version": "1.0.0",
        "model_line": "quantum-1-echelon",
        "global_step": global_step,
        "processed_tokens": processed_tokens,
        "data_offset": data_offset,
        "python_random_state": list(random.getstate()),
    }
    (directory / "trainer_state.json").write_text(
        json.dumps(state, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = build_manifest(
        directory,
        run_id=run_id,
        processed_tokens=processed_tokens,
    )
    manifest_path = directory / "checkpoint-manifest.json"
    write_atomic(manifest_path, manifest)
    return manifest_path


def restore_checkpoint(
    directory: Path,
    *,
    model: LlamaForCausalLM,
    optimizer: AdamW,
    scheduler: LambdaLR,
) -> dict[str, Any]:
    state = json.loads((directory / "trainer_state.json").read_text(encoding="utf-8"))
    model.load_state_dict(load_file(str(directory / "model.safetensors")))
    optimizer.load_state_dict(torch.load(directory / "optimizer.pt", weights_only=True))
    scheduler.load_state_dict(torch.load(directory / "scheduler.pt", weights_only=True))
    torch.set_rng_state(torch.load(directory / "torch_rng.pt", weights_only=True))
    return state


def run_smoke(
    *,
    config_path: Path,
    manifest_path: Path,
    output_dir: Path,
    run_id: str,
    max_steps: int,
    resume_dir: Path | None = None,
    learning_rate: float = 3e-4,
) -> dict[str, Any]:
    if max_steps <= 0 or max_steps > 100:
        raise ValueError("smoke max_steps must be in [1, 100]")

    config = load_yaml(config_path)
    model_cfg = config["model"]
    context_length = int(model_cfg["context_length"])
    seed = int(config.get("project", {}).get("seed", 20260922))
    random.seed(seed)
    torch.manual_seed(seed)

    hf_config = build_hf_config(model_cfg)
    hf_config.use_cache = False
    model = LlamaForCausalLM(hf_config)
    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    global_step = 0
    processed_tokens = 0
    data_offset = 0
    if resume_dir is not None:
        state = restore_checkpoint(
            resume_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        global_step = int(state["global_step"])
        processed_tokens = int(state["processed_tokens"])
        data_offset = int(state["data_offset"])

    stream = ShardedTokenStream(
        manifest_path,
        context_length=context_length,
        start_token_offset=data_offset,
        verify_files=True,
    )

    losses: list[float] = []
    steps_this_run = 0
    while steps_this_run < max_steps and stream.remaining_full_sequences > 0:
        sequence = stream.next_sequence()
        input_ids = torch.from_numpy(sequence.astype("int64", copy=True)).unsqueeze(0)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        global_step += 1
        steps_this_run += 1
        processed_tokens += context_length
        losses.append(float(loss.detach().item()))

    if steps_this_run == 0:
        raise RuntimeError("smoke run consumed zero sequences")

    checkpoint_dir = output_dir / f"checkpoint-step-{global_step}"
    manifest = save_checkpoint(
        checkpoint_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        processed_tokens=processed_tokens,
        data_offset=stream.cursor.global_token_offset,
        run_id=run_id,
    )

    result = {
        "run_id": run_id,
        "steps_this_run": steps_this_run,
        "global_step": global_step,
        "processed_tokens": processed_tokens,
        "data_offset": stream.cursor.global_token_offset,
        "losses": losses,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_manifest": str(manifest),
    }
    (output_dir / "smoke-result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded Echelon Base smoke.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    result = run_smoke(
        config_path=args.config,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        run_id=args.run_id,
        max_steps=args.max_steps,
        resume_dir=args.resume_dir,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
