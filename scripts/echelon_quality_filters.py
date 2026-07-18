#!/usr/bin/env python3

import hashlib
import re
from collections import Counter


WORD_PATTERN = re.compile(r"\w+", re.UNICODE)

BOILERPLATE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"cookie[- ]?einstellungen",
        r"datenschutzerklärung",
        r"alle rechte vorbehalten",
        r"newsletter abonnieren",
        r"javascript (?:ist )?deaktiviert",
        r"bitte akzeptieren sie die cookies",
        r"zum inhalt springen",
        r"hier kostenlos herunterladen",
        r"download(?:en)? sie (?:jetzt|hier)",
        r"copyright\s+\d{4}",
        r"melden sie sich an",
        r"passwort vergessen",
    ]
]


def words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower())


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0

    alphabetic = sum(character.isalpha() for character in text)
    return alphabetic / len(text)


def duplicate_line_ratio(text: str) -> float:
    lines = [
        " ".join(line.lower().split())
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return 0.0

    counts = Counter(lines)
    duplicate_lines = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return duplicate_lines / len(lines)


def repeated_ngram_ratio(text: str, n: int = 5) -> float:
    tokens = words(text)

    if len(tokens) < n:
        return 0.0

    ngrams = [
        tuple(tokens[index:index + n])
        for index in range(len(tokens) - n + 1)
    ]

    counts = Counter(ngrams)
    repeated = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return repeated / len(ngrams)


def boilerplate_match_count(text: str) -> int:
    return sum(
        1
        for pattern in BOILERPLATE_PATTERNS
        if pattern.search(text)
    )


def simhash64(text: str) -> int:
    tokens = words(text)

    if not tokens:
        return 0

    vector = [0] * 64

    for token, frequency in Counter(tokens).items():
        value = int.from_bytes(
            hashlib.blake2b(
                token.encode("utf-8"),
                digest_size=8,
            ).digest(),
            "big",
        )

        for bit in range(64):
            if value & (1 << bit):
                vector[bit] += frequency
            else:
                vector[bit] -= frequency

    result = 0

    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit

    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class SimHashIndex:
    def __init__(
        self,
        bits: int = 64,
        bands: int = 4,
        maximum_hamming_distance: int = 3,
    ) -> None:
        if bits <= 0:
            raise ValueError("bits muss größer als 0 sein.")

        if bands <= 0 or bits % bands != 0:
            raise ValueError("bands muss bits ohne Rest teilen.")

        self.bits = bits
        self.bands = bands
        self.maximum_hamming_distance = maximum_hamming_distance
        self.band_width = bits // bands
        self.band_mask = (1 << self.band_width) - 1

        self._buckets: list[dict[int, list[int]]] = [
            {} for _ in range(bands)
        ]

    def _band_key(self, value: int, band: int) -> int:
        shift = band * self.band_width
        return (value >> shift) & self.band_mask

    def is_near_duplicate(self, value: int) -> bool:
        candidates: set[int] = set()

        for band in range(self.bands):
            key = self._band_key(value, band)
            candidates.update(
                self._buckets[band].get(key, [])
            )

        return any(
            hamming_distance(value, candidate)
            <= self.maximum_hamming_distance
            for candidate in candidates
        )

    def add(self, value: int) -> None:
        for band in range(self.bands):
            key = self._band_key(value, band)
            self._buckets[band].setdefault(key, []).append(value)

    def __len__(self) -> int:
        if not self._buckets:
            return 0

        return sum(
            len(values)
            for values in self._buckets[0].values()
        )


def metadata_rejection(sample: dict, config: dict) -> str | None:
    filters = config["metadata_filters"]

    if sample.get("language") != filters["required_language"]:
        return "metadata_language"

    if sample.get("language_script") != filters["required_language_script"]:
        return "metadata_script"

    language_score = sample.get("language_score")
    if language_score is None or float(language_score) < filters["minimum_language_score"]:
        return "metadata_language_score"

    quality_score = sample.get("quality_score")
    if quality_score is None or float(quality_score) < filters["minimum_quality_score"]:
        return "metadata_quality_score"

    cluster_size = sample.get("minhash_cluster_size")
    if cluster_size is None or int(cluster_size) > filters["maximum_minhash_cluster_size"]:
        return "metadata_cluster_size"

    return None
