#!/usr/bin/env python3

import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset


SAMPLE_SIZE = 5000
OUTPUT = Path(
    "reports/quantum-1-echelon/source_metadata_analysis.json"
)


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "minimum": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "maximum": None,
        }

    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = round((len(ordered) - 1) * fraction)
        return float(ordered[index])

    return {
        "minimum": float(ordered[0]),
        "p05": percentile(0.05),
        "p25": percentile(0.25),
        "median": float(statistics.median(ordered)),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "maximum": float(ordered[-1]),
    }


def main() -> None:
    dataset = load_dataset(
        "epfml/FineWeb2-HQ",
        "deu_Latn",
        split="train",
        streaming=True,
    )

    languages = Counter()
    scripts = Counter()
    language_scores: list[float] = []
    quality_scores: list[float] = []
    cluster_sizes: list[int] = []
    text_lengths: list[int] = []
    missing = Counter()

    examples = []

    for index, sample in enumerate(dataset):
        if index >= SAMPLE_SIZE:
            break

        language = sample.get("language")
        script = sample.get("language_script")
        language_score = sample.get("language_score")
        quality_score = sample.get("quality_score")
        cluster_size = sample.get("minhash_cluster_size")
        text = sample.get("text") or ""

        languages[str(language)] += 1
        scripts[str(script)] += 1
        text_lengths.append(len(text))

        if language_score is None:
            missing["language_score"] += 1
        else:
            language_scores.append(float(language_score))

        if quality_score is None:
            missing["quality_score"] += 1
        else:
            quality_scores.append(float(quality_score))

        if cluster_size is None:
            missing["minhash_cluster_size"] += 1
        else:
            cluster_sizes.append(int(cluster_size))

        if len(examples) < 5:
            examples.append({
                "language": language,
                "language_script": script,
                "language_score": language_score,
                "quality_score": quality_score,
                "minhash_cluster_size": cluster_size,
                "top_langs": sample.get("top_langs"),
                "url": sample.get("url"),
                "text_preview": " ".join(text.split())[:300],
            })

    report = {
        "sample_size": sum(languages.values()),
        "languages": dict(languages),
        "language_scripts": dict(scripts),
        "language_score": quantiles(language_scores),
        "quality_score": quantiles(quality_scores),
        "minhash_cluster_size": quantiles(
            [float(value) for value in cluster_sizes]
        ),
        "text_length": quantiles(
            [float(value) for value in text_lengths]
        ),
        "missing_fields": dict(missing),
        "examples": examples,
        "note": "Embeddings wurden bewusst nicht geladen oder ausgegeben.",
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Analyse abgeschlossen.")
    print("Dokumente:", report["sample_size"])
    print("Sprachen:", report["languages"])
    print("Scripts:", report["language_scripts"])
    print("Language-Score:", report["language_score"])
    print("Quality-Score:", report["quality_score"])
    print("Clustergröße:", report["minhash_cluster_size"])
    print("Bericht:", OUTPUT)


if __name__ == "__main__":
    main()
