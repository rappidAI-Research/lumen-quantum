from scripts.train_tokenizer import DEFAULT_SPECIAL_TOKENS, load_fast_tokenizer, train_tokenizer


def test_train_and_load_sentencepiece_tokenizer_with_german_umlauts(tmp_path):
    raw_dir = tmp_path / "raw"
    tokenizer_dir = tmp_path / "tokenizer"
    raw_dir.mkdir()
    (raw_dir / "deutsch.txt").write_text(
        "Äpfel, Öl und Übermut. Grüße aus Köln.\n"
        "Das Mädchen läuft über die Straße und sagt: schön, größer, süß.\n",
        encoding="utf-8",
    )

    trained = train_tokenizer(
        input_dir=raw_dir,
        output_dir=tokenizer_dir,
        vocab_size=300,
        min_frequency=1,
        seed=7,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
    )
    loaded = load_fast_tokenizer(tokenizer_dir, DEFAULT_SPECIAL_TOKENS)

    assert (tokenizer_dir / "tokenizer.model").exists()
    assert (tokenizer_dir / "tokenizer_config.json").exists()
    assert (tokenizer_dir / "special_tokens_map.json").exists()
    assert loaded.__class__.__name__ == "SentencePieceLlamaTokenizer"
    assert len(loaded) == len(trained)

    encoded = loaded.encode("Äpfel und Grüße aus Köln", add_special_tokens=False)
    decoded = loaded.decode(encoded)

    assert encoded
    assert "Äpfel" in decoded
    assert "Grüße" in decoded
    assert "Köln" in decoded


def test_special_tokens_exist_with_llama_ids(tmp_path):
    raw_dir = tmp_path / "raw"
    tokenizer_dir = tmp_path / "tokenizer"
    raw_dir.mkdir()
    (raw_dir / "text.txt").write_text("Hallo Welt. Lumen lernt Deutsch.", encoding="utf-8")

    tokenizer = train_tokenizer(
        input_dir=raw_dir,
        output_dir=tokenizer_dir,
        vocab_size=300,
        min_frequency=1,
        seed=11,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
    )

    for token in DEFAULT_SPECIAL_TOKENS.values():
        token_id = tokenizer.convert_tokens_to_ids(token)
        assert token_id is not None

    assert tokenizer.unk_token == "<unk>"
    assert tokenizer.bos_token == "<s>"
    assert tokenizer.eos_token == "</s>"
    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.unk_token_id == 0
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.pad_token_id == 3
