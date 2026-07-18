#!/usr/bin/env python3

import json
from pathlib import Path

from datasets import load_dataset


OUTPUT = Path(
    "reports/quantum-1-echelon/quality_score_examples.json"
)

BANDS = {
    "very_low": (0.00, 0.08),
    "low": (0.08, 0.18),
    "medium": (0.18, 0.47),
    "high": (0.47, 0.94),
    "very_high": (0.94, 1.01),
}


def main() -> None:
    dataset = load_dataset(
        "epfml/FineWeb2-HQ",
        "deu_Latn",
        split="train",
        streaming=True,
    )

    examples = {name: [] for name in BANDS}

    for index, sample in enumerate(dataset):
        if index >= 5000:
            break

        score = sample.get("quality_score")
        text = sample.get("text") or ""

        if score is None:
            continue

        score = float(score)

        for name, (minimum, maximum) in BANDS.items():
            if minimum <= score < maximum and len(examples[name]) < 3:
                examples[name].append({
                    "quality_score": score,
                    "language_score": sample.get("language_score"),
                    "minhash_cluster_size": sample.get(
                        "minhash_cluster_size"
                    ),
                    "url": sample.get("url"),
                    "preview": " ".join(text.split())[:500],
                })
                break

        if all(len(values) >= 3 for values in examples.values()):
            break

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for band, values in examples.items():
        print(f"\n=== {band.upper()} ===")
        for number, item in enumerate(values, start=1):
            print(
                f"\n{number}. Score={item['quality_score']:.4f}, "
                f"Cluster={item['minhash_cluster_size']}"
            )
            print(item["preview"])


if __name__ == "__main__":
    main()
