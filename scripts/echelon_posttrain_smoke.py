"""Tiny CPU-safe SFT and DPO mechanics smoke for Quantum 1 Echelon.

The synthetic token IDs validate optimization mechanics only. They are not
training data and do not validate a future chat dataset or chat quality.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from transformers import LlamaForCausalLM

from scripts.echelon_preflight import build_hf_config


def _load_model(config_path: Path) -> LlamaForCausalLM:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hf_config = build_hf_config(config["model"])
    hf_config.use_cache = False
    torch.manual_seed(int(config.get("project", {}).get("seed", 20260922)))
    model = LlamaForCausalLM(hf_config)
    model.train()
    return model


def run_sft_smoke(config_path: Path) -> float:
    model = _load_model(config_path)
    vocab = int(model.config.vocab_size)
    length = min(16, int(model.config.max_position_embeddings))
    ids = torch.randint(4, vocab, (1, length))
    labels = ids.clone()
    labels[:, : length // 2] = -100

    optimizer = AdamW(model.parameters(), lr=1e-4)
    loss = model(input_ids=ids, labels=labels, use_cache=False).loss
    loss.backward()
    optimizer.step()
    value = float(loss.detach().item())
    if not torch.isfinite(loss):
        raise RuntimeError("SFT smoke produced non-finite loss")
    return value


def _response_logprob(
    model: LlamaForCausalLM, ids: torch.Tensor, prompt_tokens: int
) -> torch.Tensor:
    logits = model(input_ids=ids, use_cache=False).logits[:, :-1, :]
    targets = ids[:, 1:]
    token_logprobs = (
        F.log_softmax(logits, dim=-1)
        .gather(
            -1,
            targets.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    positions = torch.arange(targets.shape[1]).unsqueeze(0)
    mask = positions >= max(prompt_tokens - 1, 0)
    return (token_logprobs * mask).sum(dim=-1)


def run_dpo_smoke(config_path: Path, beta: float = 0.1) -> float:
    policy = _load_model(config_path)
    reference = copy.deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    vocab = int(policy.config.vocab_size)
    length = min(16, int(policy.config.max_position_embeddings))
    prompt_tokens = length // 2
    prompt = torch.randint(4, vocab, (1, prompt_tokens))
    chosen_tail = torch.randint(4, vocab, (1, length - prompt_tokens))
    rejected_tail = torch.randint(4, vocab, (1, length - prompt_tokens))
    chosen = torch.cat([prompt, chosen_tail], dim=1)
    rejected = torch.cat([prompt, rejected_tail], dim=1)

    with torch.no_grad():
        ref_margin = _response_logprob(reference, chosen, prompt_tokens) - _response_logprob(
            reference,
            rejected,
            prompt_tokens,
        )

    policy_margin = _response_logprob(policy, chosen, prompt_tokens) - _response_logprob(
        policy,
        rejected,
        prompt_tokens,
    )
    loss = -F.logsigmoid(beta * (policy_margin - ref_margin)).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("DPO smoke produced non-finite loss")

    optimizer = AdamW(policy.parameters(), lr=1e-5)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tiny Echelon post-training mechanics smoke.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["sft", "dpo", "both"], default="both")
    args = parser.parse_args()

    result: dict[str, Any] = {}
    if args.mode in {"sft", "both"}:
        result["sft_loss"] = run_sft_smoke(args.config)
    if args.mode in {"dpo", "both"}:
        result["dpo_loss"] = run_dpo_smoke(args.config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
