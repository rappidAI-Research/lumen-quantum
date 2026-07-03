from pathlib import Path

import torch

from scripts.train_smoke import build_llama_config, build_model, count_parameters, load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_model_initializes_with_random_weights():
    config = load_yaml_config(PROJECT_ROOT / "configs" / "smoke_5m.yaml")

    torch.manual_seed(123)
    model_a = build_model(build_llama_config(config))
    first_weight_a = next(model_a.parameters()).detach().clone()

    torch.manual_seed(456)
    model_b = build_model(build_llama_config(config))
    first_weight_b = next(model_b.parameters()).detach().clone()

    assert not torch.equal(first_weight_a, first_weight_b)


def test_forward_pass_with_example_tokens():
    config = load_yaml_config(PROJECT_ROOT / "configs" / "smoke_5m.yaml")
    llama_config = build_llama_config(config, pad_token_id=2, bos_token_id=0, eos_token_id=1)
    model = build_model(llama_config)

    input_ids = torch.randint(0, llama_config.vocab_size, (2, 16), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)
    assert outputs.logits.shape == (2, 16, llama_config.vocab_size)


def test_parameter_count_is_roughly_five_million():
    config = load_yaml_config(PROJECT_ROOT / "configs" / "smoke_5m.yaml")
    model = build_model(build_llama_config(config))
    parameters = count_parameters(model)

    assert 4_500_000 <= parameters <= 5_500_000
