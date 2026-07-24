"""Check that model cards exist and stay consistent with the project's claims.

Enforces that each canonical card has License and Limitations sections, that no
card contains an unsupported claim (open weight / trained-released Echelon), and
that the Echelon card keeps its "no trained model / no weights" boundary.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARDS_DIR = ROOT / "model_cards"
EXPECTED_CARDS = ("quantum-1-pilot.md", "quantum-1.6-pilot.md", "quantum-1-echelon.md")
REQUIRED_SUBSTRINGS = ("license", "limitation")
BANNED_CLAIMS = (
    "trained echelon model released",
    "echelon model is released",
    "echelon has been trained",
    "echelon training is complete",
    "echelon training completed",
    "quantum-1-echelon is released",
    "is open weight",
    "are open weight",
    "is open-weight",
    "are open-weight",
)
ECHELON_REQUIRED = ("no checkpoint", "model-weight license", "trained model card")


def card_errors() -> list[str]:
    errors: list[str] = []
    for name in EXPECTED_CARDS:
        path = CARDS_DIR / name
        if not path.is_file():
            errors.append(f"missing model card: model_cards/{name}")
            continue
        lowered = " ".join(path.read_text(encoding="utf-8").lower().split())
        for needle in REQUIRED_SUBSTRINGS:
            if needle not in lowered:
                errors.append(f"model_cards/{name}: missing '{needle}' section")
        for phrase in BANNED_CLAIMS:
            if phrase in lowered:
                errors.append(f"model_cards/{name}: unsupported claim '{phrase}'")
    echelon = CARDS_DIR / "quantum-1-echelon.md"
    if echelon.is_file():
        lowered = " ".join(echelon.read_text(encoding="utf-8").lower().split())
        for needle in ECHELON_REQUIRED:
            if needle not in lowered:
                errors.append(
                    f"model_cards/quantum-1-echelon.md: missing boundary phrase '{needle}'"
                )
    return errors


def main() -> int:
    errors = card_errors()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Model-card consistency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
