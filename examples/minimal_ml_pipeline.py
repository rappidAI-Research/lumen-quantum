"""Minimal, self-contained ML pipeline test (NOT a useful language model).

This example trains a tiny SentencePiece tokenizer on synthetic text, builds a
very small randomly initialized Llama-style model, and runs a few training
steps. It uses only self-generated text, downloads nothing, and is explicitly a
pipeline smoke test rather than a usable model.

Requires the optional ML dependencies:

    python -m pip install -e ".[ml]"
    python examples/minimal_ml_pipeline.py
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

REQUIRED_MODULES = ("torch", "transformers", "sentencepiece")
SYNTHETIC_CORPUS = """Lumen ist ein lokaler Testassistent.
Ein kleines Modell lernt zuerst nur den Ablauf.
Die Sonne scheint ueber der Stadt.
Reproduzierbarkeit bedeutet nachvollziehbare Schritte.
Frage: Was ist ein Smoke-Test? Antwort: Ein kurzer Funktionstest.
"""


def dependencies_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES)


def _train_tokenizer(corpus_path: Path, model_prefix: Path, vocab_size: int) -> Path:
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        hard_vocab_limit=False,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3,
    )
    return model_prefix.with_suffix(".model")


def run(steps: int = 3, vocab_size: int = 48) -> dict[str, float]:
    import sentencepiece as spm
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        corpus = workspace / "corpus.txt"
        corpus.write_text(SYNTHETIC_CORPUS, encoding="utf-8")
        tokenizer_model = _train_tokenizer(corpus, workspace / "tok", vocab_size)

        processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
        ids = processor.encode(SYNTHETIC_CORPUS, out_type=int)
        block = 16
        while len(ids) < block + 1:
            ids = ids + ids
        input_ids = torch.tensor([ids[:block]], dtype=torch.long)
        labels = torch.tensor([ids[1 : block + 1]], dtype=torch.long)

        config = LlamaConfig(
            vocab_size=processor.get_piece_size(),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=block,
            tie_word_embeddings=True,
        )
        torch.manual_seed(0)
        model = LlamaForCausalLM(config)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        losses: list[float] = []
        for _ in range(steps):
            optimizer.zero_grad()
            output = model(input_ids=input_ids, labels=labels)
            output.loss.backward()
            optimizer.step()
            losses.append(float(output.loss.detach()))
        return {"first_loss": losses[0], "last_loss": losses[-1], "steps": float(steps)}


def main() -> int:
    if not dependencies_available():
        print(
            "Optional ML dependencies are not installed. Install them with:\n"
            '    python -m pip install -e ".[ml]"\n'
            "This example is a pipeline smoke test, not a usable language model."
        )
        return 0
    metrics = run()
    print(
        "Minimal ML pipeline smoke test complete (NOT a useful model).\n"
        f"  steps: {int(metrics['steps'])}\n"
        f"  first loss: {metrics['first_loss']:.4f}\n"
        f"  last loss:  {metrics['last_loss']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
