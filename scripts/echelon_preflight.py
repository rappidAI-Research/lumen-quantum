#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from accelerate import init_empty_weights
from transformers import LlamaConfig, LlamaForCausalLM


def gib(value: int | float) -> float:
    return round(float(value) / (1024**3), 3)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Konfiguration nicht gefunden: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict) or "model" not in data or "preflight" not in data:
        raise ValueError("Die YAML-Datei benötigt die Bereiche 'model' und 'preflight'.")

    return data


def manual_parameter_count(model: dict[str, Any]) -> int:
    vocab = int(model["vocab_size"])
    hidden = int(model["hidden_size"])
    intermediate = int(model["intermediate_size"])
    layers = int(model["num_hidden_layers"])
    attention_heads = int(model["num_attention_heads"])
    kv_heads = int(model["num_key_value_heads"])

    if hidden % attention_heads != 0:
        raise ValueError("hidden_size muss durch num_attention_heads teilbar sein.")

    head_dim = hidden // attention_heads
    kv_width = kv_heads * head_dim

    embeddings = vocab * hidden

    attention = (
        hidden * hidden  # q_proj
        + hidden * kv_width  # k_proj
        + hidden * kv_width  # v_proj
        + hidden * hidden  # o_proj
    )

    mlp = (
        hidden * intermediate  # gate_proj
        + hidden * intermediate  # up_proj
        + intermediate * hidden  # down_proj
    )

    layer_norms = 2 * hidden
    transformer_layers = layers * (attention + mlp + layer_norms)
    final_norm = hidden

    lm_head = 0 if bool(model["tie_word_embeddings"]) else vocab * hidden

    return embeddings + transformer_layers + final_norm + lm_head


def build_hf_config(model: dict[str, Any]) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=int(model["vocab_size"]),
        max_position_embeddings=int(model["context_length"]),
        hidden_size=int(model["hidden_size"]),
        intermediate_size=int(model["intermediate_size"]),
        num_hidden_layers=int(model["num_hidden_layers"]),
        num_attention_heads=int(model["num_attention_heads"]),
        num_key_value_heads=int(model["num_key_value_heads"]),
        hidden_act=str(model["hidden_act"]),
        rms_norm_eps=float(model["rms_norm_eps"]),
        rope_theta=float(model["rope_theta"]),
        attention_bias=bool(model["attention_bias"]),
        mlp_bias=bool(model["mlp_bias"]),
        tie_word_embeddings=bool(model["tie_word_embeddings"]),
        initializer_range=float(model["initializer_range"]),
        use_cache=bool(model["use_cache"]),
        pad_token_id=int(model["pad_token_id"]),
        bos_token_id=int(model["bos_token_id"]),
        eos_token_id=int(model["eos_token_id"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter- und Speicher-Preflight für quantum-1-echelon."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/echelon/quantum-1-echelon-base.yaml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/quantum-1-echelon/quantum-1-echelon-base-preflight.json"),
    )
    args = parser.parse_args()

    config_data = load_config(args.config)
    model_cfg = config_data["model"]
    preflight_cfg = config_data["preflight"]

    hf_config = build_hf_config(model_cfg)

    # Das Modell wird auf dem Meta-Device aufgebaut. Dadurch werden keine
    # echten 500M-Parameter-Tensoren im RAM oder VRAM angelegt.
    with init_empty_weights():
        model = LlamaForCausalLM(hf_config)

    # Bei der Meta-Initialisierung werden geteilte Embeddings nicht immer
    # automatisch als dieselbe Parameterreferenz behandelt. Vor der Zählung
    # deshalb lm_head und Token-Embeddings ausdrücklich verknüpfen.
    model.tie_weights()

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    formula_parameters = manual_parameter_count(model_cfg)

    if formula_parameters != total_parameters:
        raise RuntimeError(
            "Parameterzählung widersprüchlich: "
            f"Formel={formula_parameters:,}, Modell={total_parameters:,}"
        )

    minimum = int(preflight_cfg["minimum_parameters"])
    maximum = int(preflight_cfg["maximum_parameters"])
    in_target_range = minimum <= total_parameters <= maximum

    # Grobe statische Speicherwerte. Aktivierungen, CUDA-Workspace,
    # Fragmentierung und Gradient-Checkpointing sind nicht enthalten.
    memory = {
        "fp32_weights_gib": gib(total_parameters * 4),
        "bf16_weights_gib": gib(total_parameters * 2),
        "bf16_gradients_gib": gib(total_parameters * 2),
        "fp32_master_weights_gib": gib(total_parameters * 4),
        "adamw_moments_gib": gib(total_parameters * 8),
        "mixed_precision_adamw_static_total_gib": gib(total_parameters * 16),
    }

    report = {
        "model_name": model_cfg["model_name"],
        "architecture": model_cfg["architecture"],
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "manual_formula_parameters": formula_parameters,
        "target_parameters": int(preflight_cfg["target_parameters"]),
        "accepted_range": {
            "minimum": minimum,
            "maximum": maximum,
        },
        "in_target_range": in_target_range,
        "architecture_config": model_cfg,
        "memory_estimates": memory,
        "memory_note": (
            "Schätzungen enthalten keine Aktivierungen, CUDA-Workspaces, "
            "Allocator-Fragmentierung oder temporären Tensoren."
        ),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("=== QUANTUM-1-ECHELON PREFLIGHT ===")
    print(f"Modell:               {model_cfg['model_name']}")
    print(f"Parameter gesamt:     {total_parameters:,}")
    print(f"Trainierbar:          {trainable_parameters:,}")
    print(f"Zielbereich:          {minimum:,} bis {maximum:,}")
    print(f"Im Zielbereich:       {in_target_range}")
    print(f"BF16-Gewichte:        {memory['bf16_weights_gib']} GiB")
    print(f"AdamW statisch ca.:  {memory['mixed_precision_adamw_static_total_gib']} GiB")
    print(f"Bericht:              {args.report}")

    if not in_target_range:
        raise SystemExit(
            f"FEHLER: {total_parameters:,} Parameter liegen außerhalb des erlaubten Bereichs."
        )


if __name__ == "__main__":
    main()
