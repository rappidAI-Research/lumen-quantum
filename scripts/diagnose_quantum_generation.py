"""Reproduzierbare Generationsdiagnose fuer quantum-1.6-pilot.

Das Skript veraendert keine Gewichte, Tokenizer, Trainingsdaten, GGUF-Dateien
oder Android-App-Artefakte. Es liest lokale Artefakte, erzeugt Diagnoseausgaben
und schreibt modellisolierte Reports unter data/diagnostics/quantum-1.6-pilot/.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import platform
import random
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import torch
import yaml
from transformers import LlamaConfig, LlamaForCausalLM

try:
    import sentencepiece as spm
except ImportError as exc:  # pragma: no cover - depends on local environment.
    raise ImportError(
        "sentencepiece ist erforderlich. Installiere: pip install -r requirements.txt"
    ) from exc

try:
    import transformers
except ImportError:  # pragma: no cover - transformers is already required by the project.
    transformers = None

try:
    from .generate_quantum import load_quantum_weights
except ImportError:
    from generate_quantum import load_quantum_weights


DEFAULT_CONFIG = "configs/quantum_1_6_diagnosis.yaml"
REPORT_FILENAMES = {
    "pytorch": "pytorch_generation_report.json",
    "roundtrip": "tokenizer_roundtrip_report.json",
    "gguf": "gguf_generation_report.json",
    "android": "android_diagnosis_report.json",
    "android_template": "android_capture_template.json",
    "summary": "diagnosis_summary.md",
}

ALLOWED_PUNCTUATION = set(".,;:!?\"'()[]{}<>/\\-+*=_%€§&@#")


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Diagnoseconfig nicht gefunden: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_parts(path: str | Path) -> tuple[str, ...]:
    return PurePosixPath(str(path).replace("\\", "/")).parts


def contains_path_part(path: str | Path, part: str) -> bool:
    return part in path_parts(path)


def resolve_output_dir(config: dict, output_dir: str | Path | None = None) -> Path:
    model_name = str(config["project"]["model_name"])
    resolved = Path(output_dir or config["paths"]["output_dir"])
    if not contains_path_part(resolved, model_name):
        raise ValueError(
            f"Diagnosepfad muss modellisoliert sein und '{model_name}' enthalten: {resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def set_fixed_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def tokenizer_model_path(tokenizer_dir: str | Path, filename: str = "tokenizer.model") -> Path:
    path = Path(tokenizer_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"tokenizer.model nicht gefunden: {path}")
    return path


def compare_tokenizer_files(
    model_dir: str | Path, tokenizer_dir: str | Path, filename: str = "tokenizer.model"
) -> dict:
    model_tokenizer = tokenizer_model_path(model_dir, filename)
    configured_tokenizer = tokenizer_model_path(tokenizer_dir, filename)
    model_hash = sha256_file(model_tokenizer)
    configured_hash = sha256_file(configured_tokenizer)
    identical = model_hash == configured_hash
    result = {
        "model_tokenizer": str(model_tokenizer),
        "configured_tokenizer": str(configured_tokenizer),
        "model_tokenizer_sha256": model_hash,
        "configured_tokenizer_sha256": configured_hash,
        "byte_identical": identical,
    }
    if not identical:
        raise ValueError(
            "Tokenizer mismatch: tokenizer.model im Modellordner und konfigurierter Tokenizer "
            f"sind nicht byte-identisch ({model_tokenizer} != {configured_tokenizer})."
        )
    return result


def validate_vocab_alignment(
    model_config: LlamaConfig | dict,
    tokenizer_vocab_size: int,
    expected_vocab_size: int | None = None,
) -> dict:
    model_vocab_size = int(
        model_config["vocab_size"] if isinstance(model_config, dict) else model_config.vocab_size
    )
    if model_vocab_size != int(tokenizer_vocab_size):
        raise ValueError(
            f"Vokabulargroesse passt nicht: config.json={model_vocab_size}, tokenizer={tokenizer_vocab_size}."
        )
    if expected_vocab_size is not None and int(tokenizer_vocab_size) != int(expected_vocab_size):
        raise ValueError(
            f"Tokenizer-vocab_size={tokenizer_vocab_size}, erwartet laut Diagnoseconfig {expected_vocab_size}."
        )
    return {
        "model_vocab_size": model_vocab_size,
        "tokenizer_vocab_size": int(tokenizer_vocab_size),
        "expected_vocab_size": expected_vocab_size,
        "ok": True,
    }


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def validate_model_shapes(
    model: LlamaForCausalLM, tokenizer_vocab_size: int, expected_parameter_count: int | None
) -> dict:
    parameter_count = count_parameters(model)
    if expected_parameter_count is not None and parameter_count != int(expected_parameter_count):
        raise ValueError(
            f"Parameterzahl stimmt nicht: {parameter_count}, erwartet {expected_parameter_count}."
        )
    embedding_shape = list(model.get_input_embeddings().weight.shape)
    lm_head_shape = list(model.get_output_embeddings().weight.shape)
    if int(embedding_shape[0]) != int(tokenizer_vocab_size):
        raise ValueError(f"Embedding-Groesse passt nicht zum Tokenizer: {embedding_shape}.")
    if int(lm_head_shape[0]) != int(tokenizer_vocab_size):
        raise ValueError(f"lm_head-Groesse passt nicht zum Tokenizer: {lm_head_shape}.")
    return {
        "parameter_count": parameter_count,
        "expected_parameter_count": expected_parameter_count,
        "embedding_shape": embedding_shape,
        "lm_head_shape": lm_head_shape,
        "ok": True,
    }


def load_sentencepiece(tokenizer_dir: str | Path) -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=str(tokenizer_model_path(tokenizer_dir)))


def tokenizer_roundtrip_for_prompts(tokenizer: Any, prompts: list[str]) -> dict:
    results = []
    stable_count = 0
    for index, prompt in enumerate(prompts):
        first_ids = list(tokenizer.encode(prompt, out_type=int))
        decoded = tokenizer.decode(first_ids)
        second_ids = list(tokenizer.encode(decoded, out_type=int))
        stable = first_ids == second_ids
        stable_count += int(stable)
        results.append(
            {
                "prompt_index": index,
                "prompt": prompt,
                "first_ids": first_ids,
                "decoded": decoded,
                "second_ids": second_ids,
                "stable": stable,
            }
        )
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "prompt_count": len(prompts),
        "stable_count": stable_count,
        "all_stable": stable_count == len(prompts),
        "results": results,
    }


def unusual_characters(text: str) -> list[dict]:
    counter = Counter(
        char
        for char in text
        if not (char.isalnum() or char.isspace() or char in ALLOWED_PUNCTUATION)
    )
    return [
        {"character": char, "codepoint": f"U+{ord(char):04X}", "count": count}
        for char, count in sorted(counter.items(), key=lambda item: (ord(item[0]), item[0]))
    ]


def eos_behavior(generated_ids: list[int], eos_token_id: int) -> dict:
    first_index = generated_ids.index(eos_token_id) if eos_token_id in generated_ids else None
    return {
        "contains_eos": first_index is not None,
        "first_eos_index": first_index,
        "ended_with_eos": bool(generated_ids and generated_ids[-1] == eos_token_id),
    }


def build_generation_record(
    *,
    prompt_index: int,
    prompt: str,
    mode_name: str,
    mode_config: dict,
    prompt_token_ids: list[int],
    generation_input_ids: list[int],
    output_token_ids: list[int],
    generated_token_ids: list[int],
    decoded_text: str,
    decoded_generated_text: str,
    unk_token_id: int,
    eos_token_id: int,
) -> dict:
    return {
        "prompt_index": prompt_index,
        "prompt": prompt,
        "mode": mode_name,
        "mode_config": mode_config,
        "prompt_token_ids": prompt_token_ids,
        "generation_input_ids": generation_input_ids,
        "output_token_ids": output_token_ids,
        "generated_token_ids": generated_token_ids,
        "decoded_text": decoded_text,
        "decoded_generated_text": decoded_generated_text,
        "unusual_characters": unusual_characters(decoded_text),
        "unknown_token_count": int(generated_token_ids.count(unk_token_id)),
        "eos_behavior": eos_behavior(generated_token_ids, eos_token_id),
    }


def load_local_model(checkpoint_dir: str | Path, device: str | None) -> LlamaForCausalLM:
    checkpoint = Path(checkpoint_dir)
    config_file = checkpoint / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Modell-Config nicht gefunden: {config_file}")
    model_config = LlamaConfig.from_json_file(str(config_file))
    model = LlamaForCausalLM(model_config)
    load_quantum_weights(model, checkpoint)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(resolved_device)
    model.eval()
    return model


def build_pytorch_preflight(
    config: dict, model: LlamaForCausalLM, sp: spm.SentencePieceProcessor
) -> dict:
    checkpoint_dir = Path(config["paths"]["pytorch_checkpoint_dir"])
    tokenizer_dir = Path(config["paths"]["tokenizer_dir"])
    tokenizer_compare = compare_tokenizer_files(
        checkpoint_dir,
        tokenizer_dir,
        config.get("expected", {}).get("tokenizer_model_file", "tokenizer.model"),
    )
    vocab = validate_vocab_alignment(
        model.config,
        int(sp.get_piece_size()),
        config.get("expected", {}).get("vocab_size"),
    )
    shapes = validate_model_shapes(
        model,
        int(sp.get_piece_size()),
        config.get("expected", {}).get("parameter_count"),
    )
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "tokenizer_file_check": tokenizer_compare,
        "vocab_check": vocab,
        "shape_check": shapes,
    }


@torch.no_grad()
def run_pytorch_diagnosis(
    config: dict,
    output_dir: Path,
    device: str | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
) -> Path:
    seed = int(config["seed"])
    set_fixed_seed(seed)
    prompts = list(config["prompts"])
    tokenizer_dir = Path(config["paths"]["tokenizer_dir"])
    sp = load_sentencepiece(tokenizer_dir)
    model = load_local_model(config["paths"]["pytorch_checkpoint_dir"], device)
    preflight = build_pytorch_preflight(config, model, sp)

    roundtrip = tokenizer_roundtrip_for_prompts(sp, prompts)
    roundtrip_path = write_json(output_dir / REPORT_FILENAMES["roundtrip"], roundtrip)

    generations: list[dict] = []
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    bos_token_id = int(
        model.config.bos_token_id if model.config.bos_token_id is not None else sp.bos_id()
    )
    eos_token_id = int(
        model.config.eos_token_id if model.config.eos_token_id is not None else sp.eos_id()
    )
    pad_token_id = int(
        model.config.pad_token_id if model.config.pad_token_id is not None else sp.pad_id()
    )
    unk_token_id = int(sp.unk_id())
    device_obj = next(model.parameters()).device

    for mode_index, (mode_key, mode_config) in enumerate(config["generation"]["modes"].items()):
        mode_name = str(mode_config.get("label", mode_key))
        for prompt_index, prompt in enumerate(prompts):
            set_fixed_seed(seed + (mode_index * 1000) + prompt_index)
            prompt_ids = list(sp.encode(prompt, out_type=int))
            generation_input_ids = [bos_token_id, *prompt_ids]
            input_tensor = torch.tensor([generation_input_ids], dtype=torch.long, device=device_obj)
            generate_kwargs = {
                "input_ids": input_tensor,
                "max_new_tokens": max_new_tokens,
                "do_sample": bool(mode_config["do_sample"]),
                "pad_token_id": pad_token_id,
                "eos_token_id": eos_token_id,
            }
            if bool(mode_config["do_sample"]):
                generate_kwargs["temperature"] = float(mode_config["temperature"])
                generate_kwargs["top_p"] = float(mode_config["top_p"])
                generate_kwargs["top_k"] = int(mode_config.get("top_k", 40))
            output = model.generate(**generate_kwargs)
            output_ids = [int(value) for value in output[0].detach().cpu().tolist()]
            generated_ids = output_ids[len(generation_input_ids) :]
            decoded_text = sp.decode(output_ids)
            decoded_generated_text = sp.decode(generated_ids) if generated_ids else ""
            generations.append(
                build_generation_record(
                    prompt_index=prompt_index,
                    prompt=prompt,
                    mode_name=mode_name,
                    mode_config={
                        "do_sample": bool(mode_config["do_sample"]),
                        "temperature": float(mode_config["temperature"]),
                        "top_p": float(mode_config["top_p"]),
                        "top_k": int(mode_config.get("top_k", 40)),
                        "max_new_tokens": max_new_tokens,
                    },
                    prompt_token_ids=prompt_ids,
                    generation_input_ids=generation_input_ids,
                    output_token_ids=output_ids,
                    generated_token_ids=generated_ids,
                    decoded_text=decoded_text,
                    decoded_generated_text=decoded_generated_text,
                    unk_token_id=unk_token_id,
                    eos_token_id=eos_token_id,
                )
            )

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_file": str(config_path),
        "model_name": config["project"]["model_name"],
        "version": config["project"].get("version"),
        "seed": seed,
        "device": str(device_obj),
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": getattr(transformers, "__version__", "unknown")
            if transformers
            else "unknown",
            "sentencepiece": getattr(spm, "__version__", "unknown"),
        },
        "safety": {
            "modified_weights": False,
            "modified_tokenizer": False,
            "modified_training_data": False,
            "modified_gguf": False,
            "modified_android_app": False,
        },
        "preflight": preflight,
        "roundtrip_report": str(roundtrip_path),
        "generations": generations,
    }
    return write_json(output_dir / REPORT_FILENAMES["pytorch"], report)


def ensure_gguf_file(path: str | Path) -> Path:
    gguf = Path(path)
    if not gguf.exists():
        raise FileNotFoundError(f"GGUF-Datei nicht gefunden: {gguf}")
    if not gguf.is_file():
        raise FileNotFoundError(f"GGUF-Pfad ist keine Datei: {gguf}")
    return gguf


def llama_cpp_version(llama_cpp_dir: str | Path) -> dict:
    root = Path(llama_cpp_dir)
    version = {"llama_cpp_dir": str(root)}
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
        version["git_head"] = result.stdout.strip() if result.returncode == 0 else None
        version["git_error"] = result.stderr.strip() if result.returncode != 0 else None
    return version


def find_llama_cli(llama_cpp_dir: str | Path) -> Path:
    root = Path(llama_cpp_dir)
    candidates = [
        root / "build" / "bin" / "llama-cli.exe",
        root / "build" / "bin" / "llama-cli",
        root / "build" / "bin" / "Release" / "llama-cli.exe",
        root / "build" / "bin" / "llama-completion.exe",
        root / "build" / "bin" / "llama-completion",
        root / "llama-cli.exe",
        root / "llama-cli",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Kein llama.cpp Inferenz-Binary gefunden. Erwartet z.B. "
        f"{root / 'build' / 'bin' / 'llama-cli'}."
    )


def build_llama_command(
    binary: Path, gguf_file: Path, prompt: str, mode_config: dict, max_new_tokens: int, seed: int
) -> list[str]:
    temperature = float(mode_config["temperature"])
    top_p = float(mode_config["top_p"])
    top_k = int(mode_config.get("top_k", 40))
    command = [
        str(binary),
        "-m",
        str(gguf_file),
        "-p",
        prompt,
        "-n",
        str(max_new_tokens),
        "--seed",
        str(seed),
        "--temp",
        str(temperature),
        "--top-p",
        str(top_p),
        "--top-k",
        str(top_k),
    ]
    if binary.name.startswith("llama-cli"):
        command.append("--no-cnv")
    return command


def run_gguf_diagnosis(
    config: dict, output_dir: Path, config_path: str | Path = DEFAULT_CONFIG
) -> Path:
    seed = int(config["seed"])
    prompts = list(config["prompts"])
    gguf_file = ensure_gguf_file(config["paths"]["gguf_file"])
    binary = find_llama_cli(config["paths"]["llama_cpp_dir"])
    version = llama_cpp_version(config["paths"]["llama_cpp_dir"])
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    generations: list[dict] = []

    for mode_index, (mode_key, mode_config) in enumerate(config["generation"]["modes"].items()):
        mode_name = str(mode_config.get("label", mode_key))
        llama_mode = dict(mode_config)
        if not bool(mode_config["do_sample"]):
            llama_mode["temperature"] = 0.0
            llama_mode["top_p"] = 1.0
        for prompt_index, prompt in enumerate(prompts):
            command_seed = seed + (mode_index * 1000) + prompt_index
            command = build_llama_command(
                binary, gguf_file, prompt, llama_mode, max_new_tokens, command_seed
            )
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
            raw_output = result.stdout
            output_text = raw_output
            if output_text.startswith(prompt):
                output_text = output_text[len(prompt) :]
            generations.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "mode": mode_name,
                    "mode_config": {
                        "temperature": float(llama_mode["temperature"]),
                        "top_p": float(llama_mode["top_p"]),
                        "top_k": int(llama_mode.get("top_k", 40)),
                        "max_new_tokens": max_new_tokens,
                        "seed": command_seed,
                    },
                    "command": command,
                    "exit_code": int(result.returncode),
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "output_text": output_text,
                    "unusual_characters": unusual_characters(result.stdout),
                }
            )

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_file": str(config_path),
        "model_name": config["project"]["model_name"],
        "version": config["project"].get("version"),
        "seed": seed,
        "gguf_file": str(gguf_file),
        "gguf_sha256": sha256_file(gguf_file),
        "llama_binary": str(binary),
        "llama_cpp_version": version,
        "safety": {
            "modified_weights": False,
            "modified_tokenizer": False,
            "modified_training_data": False,
            "modified_gguf": False,
            "modified_android_app": False,
        },
        "generations": generations,
    }
    return write_json(output_dir / REPORT_FILENAMES["gguf"], report)


def android_capture_template(
    config: dict, gguf_sha256: str | None = None, gguf_size: int | None = None
) -> dict:
    mode_examples = {}
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    seed = int(config["seed"])
    for mode_index, (mode_key, mode_config) in enumerate(config["generation"]["modes"].items()):
        mode_name = str(mode_config.get("label", mode_key))
        mode_examples[mode_name] = {
            "temperature": float(mode_config["temperature"]),
            "top_p": float(mode_config["top_p"]),
            "top_k": int(mode_config.get("top_k", 40)),
            "seed": seed + (mode_index * 1000),
            "max_tokens": max_new_tokens,
            "context_length": int(config["expected"]["context_length"]),
        }
    return {
        "format": "jsonl",
        "one_record_per_prompt_and_mode": True,
        "required_fields": list(config["android"]["required_capture_fields"]),
        "expected_model_id": config["android"]["expected_model_id"],
        "expected_ui_model_id": config["android"]["expected_ui_model_id"],
        "expected_gguf_file": config["paths"]["gguf_file"],
        "expected_gguf_sha256": gguf_sha256,
        "expected_file_size_bytes": gguf_size,
        "prompts": list(config["prompts"]),
        "sampling_by_mode": mode_examples,
        "example_record": {
            "model_id": config["android"]["expected_model_id"],
            "ui_active_model_id": config["android"]["expected_ui_model_id"],
            "local_model_path": "/data/user/0/<package>/files/models/quantum-1.6-pilot-v1.6.0-f16.gguf",
            "file_size_bytes": gguf_size,
            "sha256": gguf_sha256,
            "prompt": config["prompts"][0],
            "mode": "controlled_sampling",
            "sampling": mode_examples.get("controlled_sampling", {}),
            "raw_stream_fragments": ["Berlin", " ist", " ..."],
            "raw_stream_text": "Berlin ist ...",
            "final_ui_text": "Berlin ist ...",
        },
    }


def read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ungueltiges JSONL in {path}:{line_number}") from exc
    return records


def validate_android_capture_record(
    record: dict, config: dict, gguf_sha256: str | None, gguf_size: int | None
) -> dict:
    required = list(config["android"]["required_capture_fields"])
    missing = [field for field in required if field not in record]
    issues: list[str] = []
    if missing:
        issues.append(f"missing_fields:{','.join(missing)}")

    expected_model_id = config["android"]["expected_model_id"]
    expected_ui_model_id = config["android"]["expected_ui_model_id"]
    if record.get("model_id") != expected_model_id:
        issues.append(f"model_id_mismatch:{record.get('model_id')!r}")
    if bool(config["android"].get("require_ui_loaded_model_match", True)):
        if record.get("ui_active_model_id") != expected_ui_model_id:
            issues.append(f"ui_active_model_mismatch:{record.get('ui_active_model_id')!r}")
        if record.get("ui_active_model_id") != record.get("model_id"):
            issues.append("ui_active_model_differs_from_loaded_model")

    if gguf_sha256 and bool(config["android"].get("require_same_gguf_sha256_as_terminal", True)):
        if record.get("sha256") != gguf_sha256:
            issues.append(f"gguf_sha256_mismatch:{record.get('sha256')!r}")
    if gguf_size is not None and bool(
        config["android"].get("require_same_file_size_as_terminal", True)
    ):
        if int(record.get("file_size_bytes", -1)) != int(gguf_size):
            issues.append(f"gguf_file_size_mismatch:{record.get('file_size_bytes')!r}")

    sampling = record.get("sampling", {})
    for field in ["temperature", "top_p", "top_k", "seed", "max_tokens", "context_length"]:
        if field not in sampling:
            issues.append(f"missing_sampling:{field}")
    fragments = record.get("raw_stream_fragments")
    if not isinstance(fragments, list) or not all(isinstance(item, str) for item in fragments):
        issues.append("raw_stream_fragments_must_be_string_list")

    raw_text = record.get("raw_stream_text")
    if raw_text is None and isinstance(fragments, list):
        raw_text = "".join(fragments)

    return {
        "prompt": record.get("prompt"),
        "mode": record.get("mode"),
        "model_id": record.get("model_id"),
        "ui_active_model_id": record.get("ui_active_model_id"),
        "local_model_path": record.get("local_model_path"),
        "file_size_bytes": record.get("file_size_bytes"),
        "sha256": record.get("sha256"),
        "sampling": sampling,
        "raw_stream_text": raw_text or "",
        "issue_count": len(issues),
        "issues": issues,
    }


def run_android_diagnosis(
    config: dict, output_dir: Path, config_path: str | Path = DEFAULT_CONFIG
) -> Path:
    gguf_path = Path(config["paths"]["gguf_file"])
    gguf_sha256 = sha256_file(gguf_path) if gguf_path.exists() else None
    gguf_size = gguf_path.stat().st_size if gguf_path.exists() else None
    template = android_capture_template(config, gguf_sha256=gguf_sha256, gguf_size=gguf_size)
    template_path = write_json(output_dir / REPORT_FILENAMES["android_template"], template)

    capture_path = Path(
        config["paths"].get("android_capture_file", output_dir / "android_capture.jsonl")
    )
    if not capture_path.exists():
        report = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config_file": str(config_path),
            "status": "awaiting_android_capture",
            "android_capture_file": str(capture_path),
            "capture_template": str(template_path),
            "note": "Android-App-Log fehlt noch. Template verwenden und danach --mode android erneut ausfuehren.",
        }
        return write_json(output_dir / REPORT_FILENAMES["android"], report)

    records = read_jsonl(capture_path)
    validations = [
        validate_android_capture_record(
            record, config, gguf_sha256=gguf_sha256, gguf_size=gguf_size
        )
        for record in records
    ]
    issue_count = sum(item["issue_count"] for item in validations)
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_file": str(config_path),
        "status": "ok" if issue_count == 0 else "issues_found",
        "android_capture_file": str(capture_path),
        "capture_template": str(template_path),
        "expected_gguf_sha256": gguf_sha256,
        "expected_file_size_bytes": gguf_size,
        "record_count": len(records),
        "issue_count": issue_count,
        "records": validations,
    }
    return write_json(output_dir / REPORT_FILENAMES["android"], report)


def normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def classify_similarity(pytorch_text: str, gguf_text: str, similar_threshold: float) -> dict:
    left = normalize_for_compare(pytorch_text)
    right = normalize_for_compare(gguf_text)
    if left == right:
        label = "identisch"
        ratio = 1.0
    else:
        ratio = difflib.SequenceMatcher(a=left, b=right).ratio()
        label = "aehnlich" if ratio >= similar_threshold else "stark abweichend"
    return {"classification": label, "similarity": ratio, "pytorch_text": left, "gguf_text": right}


def index_generations(report: dict, text_key: str) -> dict[tuple[int, str], str]:
    indexed: dict[tuple[int, str], str] = {}
    for item in report.get("generations", []):
        indexed[(int(item["prompt_index"]), str(item["mode"]))] = str(item.get(text_key, ""))
    return indexed


def run_compare(config: dict, output_dir: Path) -> Path:
    pytorch_path = output_dir / REPORT_FILENAMES["pytorch"]
    gguf_path = output_dir / REPORT_FILENAMES["gguf"]
    if not pytorch_path.exists():
        raise FileNotFoundError(
            f"PyTorch-Report fehlt: {pytorch_path}. Fuehre zuerst --mode pytorch aus."
        )
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF-Report fehlt: {gguf_path}. Fuehre zuerst --mode gguf aus.")

    pytorch_report = read_json(pytorch_path)
    gguf_report = read_json(gguf_path)
    pytorch_items = index_generations(pytorch_report, "decoded_text")
    gguf_items = index_generations(gguf_report, "output_text")
    android_path = output_dir / REPORT_FILENAMES["android"]
    android_items: dict[tuple[int, str], str] = {}
    android_status = "not_available"
    if android_path.exists():
        android_report = read_json(android_path)
        android_status = str(android_report.get("status", "unknown"))
        for record in android_report.get("records", []):
            try:
                prompt_index = list(config["prompts"]).index(record.get("prompt"))
            except ValueError:
                continue
            mode = str(record.get("mode"))
            android_items[(prompt_index, mode)] = str(record.get("raw_stream_text", ""))
    similar_threshold = float(config.get("compare", {}).get("similar_threshold", 0.72))

    rows = []
    counts = Counter()
    all_keys = sorted(set(pytorch_items) | set(gguf_items))
    prompts = list(config["prompts"])
    for prompt_index, mode in all_keys:
        comparison = classify_similarity(
            pytorch_items.get((prompt_index, mode), ""),
            gguf_items.get((prompt_index, mode), ""),
            similar_threshold,
        )
        counts[comparison["classification"]] += 1
        rows.append(
            {
                "prompt_index": prompt_index,
                "prompt": prompts[prompt_index] if prompt_index < len(prompts) else "",
                "mode": mode,
                **comparison,
            }
        )

    android_rows = []
    android_counts = Counter()
    for key in sorted(set(gguf_items) & set(android_items)):
        prompt_index, mode = key
        comparison = classify_similarity(gguf_items[key], android_items[key], similar_threshold)
        android_counts[comparison["classification"]] += 1
        android_rows.append(
            {
                "prompt_index": prompt_index,
                "prompt": prompts[prompt_index] if prompt_index < len(prompts) else "",
                "mode": mode,
                **comparison,
            }
        )

    lines = [
        "# quantum-1.6-pilot Diagnosezusammenfassung",
        "",
        f"- Erstellt UTC: `{datetime.now(UTC).isoformat()}`",
        f"- PyTorch-Report: `{pytorch_path}`",
        f"- GGUF-Report: `{gguf_path}`",
        f"- Android-Report: `{android_path if android_path.exists() else 'nicht vorhanden'}`",
        f"- Android-Status: `{android_status}`",
        f"- Vergleichseintraege: `{len(rows)}`",
        f"- Identisch: `{counts['identisch']}`",
        f"- Aehnlich: `{counts['aehnlich']}`",
        f"- Stark abweichend: `{counts['stark abweichend']}`",
        "",
        "Diese Zusammenfassung behauptet keine Ursache ohne Messung. Sie markiert nur die gemessene Textnaehe zwischen PyTorch- und GGUF-Ausgaben.",
        "",
        "| Prompt | Modus | Klasse | Aehnlichkeit |",
        "|---|---|---|---:|",
    ]
    for row in rows:
        prompt = row["prompt"].replace("\n", "\\n")
        lines.append(
            f"| {prompt} | {row['mode']} | {row['classification']} | {row['similarity']:.3f} |"
        )

    if android_rows:
        lines.extend(
            [
                "",
                "## GGUF-Terminal vs. Android",
                "",
                f"- Identisch: `{android_counts['identisch']}`",
                f"- Aehnlich: `{android_counts['aehnlich']}`",
                f"- Stark abweichend: `{android_counts['stark abweichend']}`",
                "",
                "| Prompt | Modus | Klasse | Aehnlichkeit |",
                "|---|---|---|---:|",
            ]
        )
        for row in android_rows:
            prompt = row["prompt"].replace("\n", "\\n")
            lines.append(
                f"| {prompt} | {row['mode']} | {row['classification']} | {row['similarity']:.3f} |"
            )

    summary = "\n".join(lines) + "\n"
    path = output_dir / REPORT_FILENAMES["summary"]
    path.write_text(summary, encoding="utf-8")

    comparison_json = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pytorch_report": str(pytorch_path),
        "gguf_report": str(gguf_path),
        "counts": dict(counts),
        "comparisons": rows,
        "android_status": android_status,
        "terminal_vs_android_counts": dict(android_counts),
        "terminal_vs_android_comparisons": android_rows,
    }
    write_json(output_dir / "diagnosis_comparison.json", comparison_json)
    return path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostiziert quantum-1.6-pilot Generationen.")
    parser.add_argument(
        "--mode", choices=["pytorch", "gguf", "android", "compare", "all"], default="all"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=["cpu", "cuda"], help="PyTorch-Zielgeraet.")
    parser.add_argument("--output-dir", help="Override fuer modellisolierten Diagnoseordner.")
    parser.add_argument(
        "--llama-cpp-dir",
        help="Expliziter Pfad zu einem externen llama.cpp-Checkout fuer GGUF-Modi.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.llama_cpp_dir:
        config["paths"]["llama_cpp_dir"] = args.llama_cpp_dir
    if args.mode in {"gguf", "all"} and not config["paths"].get("llama_cpp_dir"):
        raise ValueError(
            "--llama-cpp-dir ist fuer GGUF-Diagnosen erforderlich. Siehe docs/gguf-export.md."
        )
    output_dir = resolve_output_dir(config, args.output_dir)

    if args.mode in {"pytorch", "all"}:
        path = run_pytorch_diagnosis(
            config, output_dir, device=args.device, config_path=args.config
        )
        print(f"PyTorch-Report: {path}")
    if args.mode in {"gguf", "all"}:
        path = run_gguf_diagnosis(config, output_dir, config_path=args.config)
        print(f"GGUF-Report: {path}")
    if args.mode in {"android", "all"}:
        path = run_android_diagnosis(config, output_dir, config_path=args.config)
        print(f"Android-Diagnosereport: {path}")
    if args.mode in {"compare", "all"}:
        path = run_compare(config, output_dir)
        print(f"Diagnosezusammenfassung: {path}")


if __name__ == "__main__":
    main()
