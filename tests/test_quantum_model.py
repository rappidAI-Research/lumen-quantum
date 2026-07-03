import json
from pathlib import Path

import pytest
import sentencepiece as spm
import torch
import yaml
from accelerate import Accelerator
from transformers import LlamaConfig, LlamaForCausalLM

from scripts.generate_quantum import load_quantum_weights
from scripts.inspect_model_size import (
    build_quantum_llama_config,
    build_quantum_model,
    count_parameters,
    inspect_model_size,
    load_quantum_tokenizer_info,
)
from scripts.train_quantum_pilot import checkpoint_for_step, checkpoint_is_complete, save_checkpoint


def _write_synthetic_corpus(path: Path) -> None:
    syllables = [
        "lum",
        "quant",
        "spr",
        "ach",
        "modell",
        "daten",
        "licht",
        "kern",
        "fluss",
        "raum",
        "zeit",
        "wort",
        "satz",
        "klar",
        "ruh",
        "deut",
        "sch",
        "lern",
        "blick",
        "form",
        "wert",
        "ziel",
        "pfad",
        "code",
        "test",
        "haus",
        "stadt",
        "baum",
        "berg",
        "feld",
    ]
    words: list[str] = []
    for i, first in enumerate(syllables):
        for j, second in enumerate(syllables):
            for k, third in enumerate(syllables):
                words.append(f"{first}{second}{third}{i:02d}{j:02d}{k:02d}")
                if len(words) >= 24000:
                    break
            if len(words) >= 24000:
                break
        if len(words) >= 24000:
            break

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(0, len(words), 40):
            handle.write(
                "Äpfel Öl Grüße Straße Fußgänger Lumen deutscher Tokenizer "
                + " ".join(words[index : index + 40])
                + "\n"
            )


@pytest.fixture(scope="module")
def quantum_model_config(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("quantum_model")
    tokenizer_dir = root / "tokenizer" / "quantum-1-pilot"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    corpus = root / "corpus.txt"
    _write_synthetic_corpus(corpus)

    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(tokenizer_dir / "tokenizer"),
        model_type="bpe",
        vocab_size=16384,
        character_coverage=1.0,
        hard_vocab_limit=True,
        byte_fallback=False,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="nfkc",
        user_defined_symbols="<|system|>,<|user|>,<|assistant|>",
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        minloglevel=1,
    )
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_dir / "tokenizer.model"))
    token_ids = {token: int(sp.piece_to_id(token)) for token in ["<unk>", "<s>", "</s>", "<pad>", "<|system|>", "<|user|>", "<|assistant|>"]}
    (tokenizer_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "LlamaTokenizer",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"],
                "add_bos_token": True,
                "add_eos_token": False,
                "legacy": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (tokenizer_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (tokenizer_dir / "tokenizer_manifest.json").write_text(
        json.dumps(
            {
                "tokenizer_name": "quantum-1-pilot-test",
                "training_file": str(corpus),
                "training_data_sha256": "test",
                "requested_vocab_size": 16384,
                "actual_vocab_size": 16384,
                "future_llama_vocab_size": 16384,
                "token_ids": token_ids,
                "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"],
                "sentencepiece_version": getattr(spm, "__version__", "unknown"),
                "seed": 123,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    config = {
        "project": {"name": "Lumen Quantum", "model_name": "quantum-1-base"},
        "seed": 123,
        "tokenizer": {"dir": str(tokenizer_dir), "manifest_file": "tokenizer_manifest.json"},
        "data": {
            "train_file": str(root / "train.jsonl"),
            "validation_file": str(root / "validation.jsonl"),
            "test_file": str(root / "test.jsonl"),
            "text_field": "text",
            "block_size": 512,
        },
        "model": {
            "architecture": "LlamaForCausalLM",
            "vocab_size": "auto",
            "hidden_size": 512,
            "intermediate_size": 1536,
            "num_hidden_layers": 12,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "max_position_embeddings": 512,
            "rms_norm_eps": 0.000001,
            "rope_theta": 10000.0,
            "tie_word_embeddings": True,
            "attention_dropout": 0.0,
            "initializer_range": 0.02,
            "parameter_count_min": 45000000,
            "parameter_count_max": 55000000,
        },
        "training": {"output_dir": str(root / "models" / "quantum-1-base")},
    }
    config_path = root / "quantum_1_base_pilot.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config_path


def test_quantum_base_parameter_count_is_in_target_range(quantum_model_config):
    report = inspect_model_size(quantum_model_config)

    assert report["vocab_size"] == 16384
    assert 45_000_000 <= report["parameter_count"] <= 55_000_000
    assert report["parameter_count"] == 49_295_872


def test_quantum_forward_pass_and_tokenizer_compatibility(quantum_model_config):
    config = yaml.safe_load(Path(quantum_model_config).read_text(encoding="utf-8"))
    tokenizer_info = load_quantum_tokenizer_info(config)
    llama_config = build_quantum_llama_config(config, tokenizer_info)
    model = build_quantum_model(llama_config)

    assert llama_config.vocab_size == tokenizer_info.vocab_size
    assert llama_config.bos_token_id == 1
    assert llama_config.eos_token_id == 2
    assert llama_config.pad_token_id == 3
    assert getattr(llama_config, "unk_token_id") == 0

    input_ids = torch.randint(0, llama_config.vocab_size, (1, 8), dtype=torch.long)
    outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=input_ids)

    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)
    assert outputs.logits.shape == (1, 8, llama_config.vocab_size)


def test_quantum_model_initializes_randomly(quantum_model_config):
    config = yaml.safe_load(Path(quantum_model_config).read_text(encoding="utf-8"))
    tokenizer_info = load_quantum_tokenizer_info(config)
    llama_config = build_quantum_llama_config(config, tokenizer_info)

    torch.manual_seed(1)
    first_model = build_quantum_model(llama_config)
    first_weight = next(first_model.parameters()).detach().clone()
    del first_model

    torch.manual_seed(2)
    second_model = build_quantum_model(llama_config)
    second_weight = next(second_model.parameters()).detach().clone()

    assert not torch.equal(first_weight, second_weight)


def test_quantum_model_saves_and_loads_locally(tmp_path):
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config)
    save_dir = tmp_path / "local_model"
    model.save_pretrained(save_dir, safe_serialization=True)

    reloaded_config = LlamaConfig.from_json_file(str(save_dir / "config.json"))
    reloaded = LlamaForCausalLM(reloaded_config)
    load_quantum_weights(reloaded, save_dir)

    assert count_parameters(reloaded) == count_parameters(model)
    input_ids = torch.randint(0, config.vocab_size, (1, 4), dtype=torch.long)
    outputs = reloaded(input_ids=input_ids, labels=input_ids)
    assert torch.isfinite(outputs.loss)


def test_quantum_checkpoint_contains_full_resume_state(tmp_path):
    output_dir = tmp_path / "models" / "quantum-1-base"
    tokenizer_dir = tmp_path / "tokenizer" / "quantum-1-pilot"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "tokenizer.model").write_bytes(b"fake-model-for-copy-test")
    (tokenizer_dir / "tokenizer.vocab").write_text("<unk>\t0\n", encoding="utf-8")
    (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tokenizer_dir / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (tokenizer_dir / "tokenizer_manifest.json").write_text(
        json.dumps({"tokenizer_name": "quantum-1-pilot-test", "actual_vocab_size": 128}),
        encoding="utf-8",
    )

    accelerator = Accelerator(cpu=True)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    input_ids = torch.randint(0, config.vocab_size, (1, 8), dtype=torch.long)
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        global_step=1,
        epoch=1,
        config={"project": {"model_name": "quantum-1-base"}},
        tokenizer_dir=tokenizer_dir,
    )

    checkpoint = checkpoint_for_step(output_dir, 1)
    assert checkpoint_is_complete(checkpoint)
    assert (checkpoint / "model.safetensors").exists()
    assert (checkpoint / "tokenizer_manifest.json").exists()
    assert (checkpoint / "tokenizer" / "tokenizer_manifest.json").exists()

    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["global_step"] == 1
    assert state["epoch"] == 1
    assert "optimizer" in state
    assert "scheduler" in state
    assert "rng_state" in state
    assert "config" in state
    assert state["tokenizer_manifest"]["tokenizer_name"] == "quantum-1-pilot-test"
