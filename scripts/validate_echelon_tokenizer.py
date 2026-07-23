#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def load_test_cases():
    from tests.echelon.test_tokenizer_cases import TEST_CASES

    return TEST_CASES


def load_tokenizer(path: Path):
    """
    Wird aktiviert, sobald der echte Echelon-Tokenizer existiert.
    Erwartet SentencePiece-kompatible Tokenizerdateien.
    """
    import sentencepiece as spm

    model_files = list(path.glob("*.model"))

    if not model_files:
        raise FileNotFoundError(f"Kein Tokenizer-Modell gefunden in {path}")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(model_files[0]))

    return tokenizer


def validate_roundtrip(tokenizer, text: str):
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    return {
        "input": text,
        "tokens": len(ids),
        "decoded": decoded,
        "exact_match": decoded == text,
    }


def main():
    tokenizer_path = Path("tokenizer/echelon")

    if not list(tokenizer_path.glob("*.model")):
        print("INFO: Noch kein trainierter Echelon-Tokenizer vorhanden.")
        print("Die Teststruktur ist vorbereitet.")
        return

    tokenizer = load_tokenizer(tokenizer_path)

    results = []

    for case in load_test_cases():
        results.append(validate_roundtrip(tokenizer, case))

    failed = [item for item in results if not item["exact_match"]]

    report = {
        "total_cases": len(results),
        "failed_cases": len(failed),
        "results": results,
    }

    output = Path("reports/quantum-1-echelon/tokenizer_validation.json")

    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if failed:
        print(f"FEHLER: {len(failed)} Roundtrip-Probleme")
        sys.exit(1)

    print(f"Tokenizer-Test bestanden: {len(results)} Fälle")


if __name__ == "__main__":
    main()
