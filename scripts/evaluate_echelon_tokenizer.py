#!/usr/bin/env python3

from pathlib import Path
import json
import sentencepiece as spm


TEST_TEXTS = {
    "german_general":
        "Die künstliche Intelligenz verändert unsere Gesellschaft und Technologie.",

    "german_umlauts":
        "Äpfel, Öl, Überprüfung, Größe, Straße, Fußball, süße Grüße.",

    "english":
        "Artificial intelligence is changing the future of technology.",

    "code":
        """
def calculate_sum(values):
    return sum(values)
        """,

    "markdown":
        """
# Überschrift

- Punkt eins
- Punkt zwei
        """,

    "json":
        '{"model":"quantum-1-echelon","parameters":500000000}',

    "chat":
        """
<|system|>
Du bist ein hilfreicher Assistent.
<|end|>

<|user|>
Erkläre neuronale Netze.
<|end|>

<|assistant|>
Neuronale Netze sind Modelle des maschinellen Lernens.
<|end|>
        """,

    "unicode":
        "😀 🚀 🧠 café naïve İstanbul 東京 Москва",
}


def load_tokenizer():
    path = Path(
        "tokenizer/echelon/"
        "quantum-1-echelon.model"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Kein trainierter Echelon-Tokenizer gefunden."
        )

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(path))

    return tokenizer


def main():

    tokenizer = load_tokenizer()

    results = {}

    for name, text in TEST_TEXTS.items():

        tokens = tokenizer.encode(text)

        results[name] = {
            "characters": len(text),
            "tokens": len(tokens),
            "tokens_per_character":
                round(
                    len(tokens) / max(len(text), 1),
                    4,
                ),
        }

    output = Path(
        "reports/quantum-1-echelon/"
        "tokenizer_quality.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
