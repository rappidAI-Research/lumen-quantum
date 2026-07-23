#!/usr/bin/env python3

from pathlib import Path

import sentencepiece as spm
import yaml


def load_config():
    with open("configs/echelon/tokenizer.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():

    config = load_config()

    input_file = Path("data/echelon/tokenizer_training/training_text.txt")

    output_dir = Path(config["output"]["directory"])

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Keine Trainingsdatei gefunden: {input_file}")

    prefix = str(output_dir / config["output"]["model_prefix"])

    print("Starte Tokenizer-Training")
    print("Input:", input_file)
    print("Output:", prefix)

    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=prefix,
        vocab_size=config["tokenizer"]["vocab_size"],
        model_type="bpe",
        character_coverage=config["tokenizer"]["character_coverage"],
        byte_fallback=config["tokenizer"]["byte_fallback"],
        split_digits=config["tokenizer"]["split_digits"],
        normalization_rule_name=config["tokenizer"]["normalization_rule_name"],
        allow_whitespace_only_pieces=config["tokenizer"]["allow_whitespace_only_pieces"],
        remove_extra_whitespaces=config["tokenizer"]["remove_extra_whitespaces"],
        hard_vocab_limit=False,
        pad_id=config["expected_ids"]["pad_token_id"],
        bos_id=config["expected_ids"]["bos_token_id"],
        eos_id=config["expected_ids"]["eos_token_id"],
        unk_id=config["expected_ids"]["unk_token_id"],
        user_defined_symbols=[
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|end|>",
        ],
    )

    print("Tokenizer erfolgreich erstellt.")


if __name__ == "__main__":
    main()
