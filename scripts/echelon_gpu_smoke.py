#!/usr/bin/env python3

from pathlib import Path

import torch
from transformers import LlamaForCausalLM

from echelon_preflight import build_hf_config, load_config


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("FEHLER: CUDA ist nicht verfügbar.")

    config = load_config(
        Path("configs/echelon/quantum-1-echelon-base.yaml")
    )
    hf_config = build_hf_config(config["model"])
    hf_config.use_cache = False

    torch.manual_seed(20260718)
    torch.cuda.manual_seed_all(20260718)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("GPU:", torch.cuda.get_device_name(0))

    model = LlamaForCausalLM(hf_config).to(
        device="cuda",
        dtype=torch.bfloat16,
    )
    model.train()

    input_ids = torch.randint(
        0,
        hf_config.vocab_size,
        (1, 128),
        device="cuda",
    )

    loss = model(
        input_ids=input_ids,
        labels=input_ids,
        use_cache=False,
    ).loss

    loss.backward()
    torch.cuda.synchronize()

    print("Loss:", round(loss.item(), 4))
    print(
        "Maximaler VRAM:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "GiB",
    )
    print("GPU-Smoke-Test: BESTANDEN")


if __name__ == "__main__":
    main()
