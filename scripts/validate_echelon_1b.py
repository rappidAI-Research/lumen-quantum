"""Validate the planning contract for Quantum 1 Echelon 1B.

This is intentionally CPU-only and network-free. It validates the committed
planning/configuration invariants before any dataset preparation or paid GPU
work is allowed to begin.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "echelon" / "1b"


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.relative_to(ROOT)} must contain a mapping at the root")
    return cast("dict[str, Any]", payload)


def manual_parameter_count(model: dict[str, Any]) -> int:
    vocab = int(model["vocab_size"])
    hidden = int(model["hidden_size"])
    intermediate = int(model["intermediate_size"])
    layers = int(model["num_hidden_layers"])
    heads = int(model["num_attention_heads"])
    kv_heads = int(model["num_key_value_heads"])

    if hidden % heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if heads % kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    head_dim = hidden // heads
    kv_width = kv_heads * head_dim
    embeddings = vocab * hidden
    attention = (2 * hidden * hidden) + (2 * hidden * kv_width)
    mlp = 3 * hidden * intermediate
    norms = 2 * hidden
    final_norm = hidden
    lm_head = 0 if bool(model["tie_word_embeddings"]) else vocab * hidden
    return embeddings + layers * (attention + mlp + norms) + final_norm + lm_head


def validate_model(path: Path) -> list[str]:
    data = load_yaml(path)
    model = data["model"]
    preflight = data["preflight"]
    problems: list[str] = []

    count = manual_parameter_count(model)
    expected = int(preflight["target_parameters"])
    minimum = int(preflight["minimum_parameters"])
    maximum = int(preflight["maximum_parameters"])

    if model["model_name"] != "quantum-1-echelon-base":
        problems.append(f"{path.name}: model_name must stay quantum-1-echelon-base")
    if int(model["context_length"]) != 4096:
        problems.append(f"{path.name}: base context_length must be 4096")
    if str(model["dtype_target"]).lower() != "bf16":
        problems.append(f"{path.name}: dtype_target must be bf16")
    if not bool(model["tie_word_embeddings"]):
        problems.append(f"{path.name}: embeddings must remain tied for the candidates")
    if count != expected:
        problems.append(f"{path.name}: formula count {count} != target_parameters {expected}")
    if not minimum <= count <= maximum:
        problems.append(f"{path.name}: parameter count {count} outside [{minimum}, {maximum}]")
    return problems


def validate_sources(*, require_production_ready: bool = False) -> list[str]:
    data = load_yaml(CONFIG_ROOT / "sources.yaml")
    sources = data.get("sources", [])
    shares = [float(item["target_share"]) for item in sources]
    problems: list[str] = []
    if abs(sum(shares) - 1.0) > 1e-9:
        problems.append(f"sources.yaml: target shares sum to {sum(shares)}, expected 1.0")

    source_ids = [str(item.get("id")) for item in sources]
    if len(source_ids) != len(set(source_ids)):
        problems.append("sources.yaml: source ids must be unique")

    for item in sources:
        source_id = str(item.get("id"))
        if item.get("review_status") is None or item.get("terms_status") is None:
            problems.append(f"sources.yaml: {source_id} must expose review/terms status")
        if item.get("removal_status") is None:
            problems.append(f"sources.yaml: {source_id} must expose removal status")
        if not isinstance(item.get("production_approved"), bool):
            problems.append(f"sources.yaml: {source_id} must expose boolean production_approved")

        dataset = item.get("dataset")
        revision = item.get("revision")
        if dataset is not None:
            if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
                problems.append(
                    f"sources.yaml: {source_id} identified dataset must pin a 40-hex revision"
                )
            evidence = item.get("evidence")
            if not isinstance(evidence, dict):
                problems.append(f"sources.yaml: {source_id} identified dataset needs evidence")
            else:
                for key in ("dataset_card", "license_metadata", "upstream_terms", "reviewed_at"):
                    if not evidence.get(key):
                        problems.append(
                            f"sources.yaml: {source_id} evidence.{key} must be non-empty"
                        )

        if item.get("production_approved") is True:
            if dataset is None:
                problems.append(f"sources.yaml: {source_id} cannot be approved without a dataset")
            if item.get("review_status") != "approved":
                problems.append(
                    f"sources.yaml: {source_id} approved source must have review_status=approved"
                )
            if item.get("terms_status") != "approved_for_project":
                problems.append(
                    f"sources.yaml: {source_id} approved source must have terms_status=approved_for_project"
                )
            if item.get("removal_status") != "ready":
                problems.append(
                    f"sources.yaml: {source_id} approved source must have removal_status=ready"
                )

        if require_production_ready and item.get("production_approved") is not True:
            problems.append(f"sources.yaml: {source_id} is not production approved")
    return problems


def validate_garden() -> list[str]:
    data = load_yaml(CONFIG_ROOT / "garden-v2.yaml")
    targets = data["targets"]
    mix = data["mix"]
    pipeline = data["pipeline"]
    problems: list[str] = []

    if int(targets["train_tokens"]) != 40_000_000_000:
        problems.append("garden-v2.yaml: fixed base target must be 40B tokens")
    if int(targets["conditional_stretch_train_tokens"]) != 50_000_000_000:
        problems.append("garden-v2.yaml: conditional stretch target must be 50B tokens")
    if int(targets["context_length"]) != 4096:
        problems.append("garden-v2.yaml: context_length must be 4096")
    if int(targets["tokens_per_shard"]) != 100_000_000:
        problems.append("garden-v2.yaml: shard size must be 100M tokens")
    if abs(sum(float(value) for value in mix.values()) - 1.0) > 1e-9:
        problems.append("garden-v2.yaml: mix shares must sum to 1.0")
    for key in (
        "pii_sensitive_filtering",
        "exact_deduplication",
        "benchmark_decontamination",
        "stable_splits",
        "manifests_required",
        "shard_checksums_required",
    ):
        if pipeline.get(key) is not True:
            problems.append(f"garden-v2.yaml: {key} must be true")
    return problems


def validate_training() -> list[str]:
    base = load_yaml(CONFIG_ROOT / "train-base.yaml")
    runtime = load_yaml(CONFIG_ROOT / "runtime-aws.yaml")
    sft = load_yaml(CONFIG_ROOT / "sft.yaml")
    dpo = load_yaml(CONFIG_ROOT / "dpo.yaml")
    problems: list[str] = []

    milestones = [int(value) for value in base["milestones_tokens"]]
    required = [
        100_000_000,
        1_000_000_000,
        5_000_000_000,
        10_000_000_000,
        20_000_000_000,
        30_000_000_000,
        40_000_000_000,
    ]
    if milestones != required:
        problems.append("train-base.yaml: milestone sequence does not match the masterplan")
    if base["runtime"].get("client_disconnect_safe") is not True:
        problems.append("train-base.yaml: client_disconnect_safe must be true")
    if base["runtime"].get("non_interactive") is not True:
        problems.append("train-base.yaml: production training must be non-interactive")
    if base["checkpoints"].get("verified_external_recovery_required") is not True:
        problems.append("train-base.yaml: verified external recovery is mandatory")

    budget = runtime["budget"]
    planned = sum(
        int(budget[key])
        for key in (
            "planned_data_cpu_storage_usd",
            "planned_base_usd",
            "protected_chat_usd",
            "protected_reserve_usd",
        )
    )
    if int(budget["promotional_credits_usd"]) != 1140:
        problems.append("runtime-aws.yaml: credit budget must be $1,140")
    if planned != int(budget["promotional_credits_usd"]):
        problems.append(f"runtime-aws.yaml: planned budget sums to ${planned}, expected $1,140")
    if budget.get("private_overage_allowed") is not False:
        problems.append("runtime-aws.yaml: private overage must remain disabled")

    execution = runtime["execution"]
    for key in (
        "client_disconnect_safe",
        "non_interactive",
        "persistent_logs",
        "machine_readable_status",
        "resume_latest_required",
    ):
        if execution.get(key) is not True:
            problems.append(f"runtime-aws.yaml: execution.{key} must be true")

    if sft["project"].get("status") != "planning":
        problems.append("sft.yaml: SFT must remain planning until data/hyperparameters are frozen")
    if dpo["project"].get("status") != "planning":
        problems.append("dpo.yaml: DPO must remain planning until SFT evaluation is complete")
    if dpo["release_gate"].get("discard_preference_checkpoint_if_worse") is not True:
        problems.append("dpo.yaml: worse preference checkpoints must be discardable")
    return problems


def validate(*, require_production_sources: bool = False) -> list[str]:
    problems: list[str] = []
    problems.extend(validate_model(CONFIG_ROOT / "model-32k.yaml"))
    problems.extend(validate_model(CONFIG_ROOT / "model-48k.yaml"))
    problems.extend(validate_sources(require_production_ready=require_production_sources))
    problems.extend(validate_garden())
    problems.extend(validate_training())
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Quantum 1 Echelon 1B planning invariants."
    )
    parser.add_argument(
        "--require-production-sources",
        action="store_true",
        help="Fail until every Garden v2 source has explicit production approval.",
    )
    args = parser.parse_args()
    problems = validate(require_production_sources=args.require_production_sources)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        return 1
    print("Quantum 1 Echelon 1B planning invariants: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
