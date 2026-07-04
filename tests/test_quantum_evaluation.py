import json
from pathlib import Path

import yaml

from scripts.evaluate_quantum import default_evaluation_config, read_completion_prompts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_final_train_config_contains_robust_evaluation_section():
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "quantum_1_final_train.yaml").read_text(encoding="utf-8"))
    evaluation = config["evaluation"]

    assert Path(evaluation["checkpoint_dir"]) == Path("models/quantum-1-base/final")
    assert evaluation["tokenizer_dir"] == "tokenizer/quantum-1"
    assert evaluation["validation_file"] == "data/quantum/final/tokenized/validation.pt"
    assert evaluation["eval_file"] == "data/evals/quantum_1_base_v1.jsonl"
    assert evaluation["output_dir"] == "data/evals/results/quantum-1-base"
    assert evaluation["validation_batch_size"] > 0
    assert evaluation["generate_samples"] is True


def test_evaluation_defaults_do_not_require_evaluation_key():
    config = {
        "training": {"output_dir": "models/quantum-1-base", "batch_size": 4},
        "tokenizer": {"dir": "tokenizer/quantum-1"},
        "data": {
            "tokenized_dir": "data/quantum/final/tokenized",
            "validation_file": "data/quantum/final/tokenized/validation.pt",
        },
    }

    evaluation = default_evaluation_config(config)

    assert Path(evaluation["checkpoint_dir"]) == Path("models/quantum-1-base/final")
    assert evaluation["tokenizer_dir"] == "tokenizer/quantum-1"
    assert evaluation["validation_file"] == "data/quantum/final/tokenized/validation.pt"
    assert evaluation["validation_batch_size"] == 4
    assert evaluation["max_new_tokens"] == 80


def test_completion_eval_file_is_jsonl_with_prompts():
    prompts = read_completion_prompts(PROJECT_ROOT / "data" / "evals" / "quantum_1_base_v1.jsonl")

    assert len(prompts) >= 6
    assert all(prompt["prompt"].strip() for prompt in prompts)
    assert {prompt["category"] for prompt in prompts} == {"completion"}


def test_read_completion_prompts_rejects_jsonl_without_prompt(tmp_path):
    eval_file = tmp_path / "bad.jsonl"
    eval_file.write_text(json.dumps({"text": "kein Prompt"}, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        read_completion_prompts(eval_file)
    except ValueError as exc:
        assert "prompt" in str(exc)
    else:  # pragma: no cover - makes the failure message clearer than bare assert False.
        raise AssertionError("JSONL ohne prompt-Feld wurde nicht abgelehnt.")
