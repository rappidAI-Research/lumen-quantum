from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_architecture_config() -> None:
    path = ROOT / "configs/echelon/quantum-1-echelon-base.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = data["model"]

    assert model["model_name"] == "quantum-1-echelon-base"
    assert model["vocab_size"] == 32768
    assert model["context_length"] == 2048
    assert model["hidden_size"] == 1280
    assert model["num_hidden_layers"] == 26
    assert model["hidden_size"] % model["num_attention_heads"] == 0
    assert model["num_attention_heads"] % model["num_key_value_heads"] == 0


def test_paths_are_isolated() -> None:
    path = ROOT / "configs/echelon/paths.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    combined = "\n".join(data["paths"].values())

    assert "quantum-1-echelon" in combined
    assert "quantum-1-base" not in combined
    assert "quantum-1.6-pilot" not in combined
