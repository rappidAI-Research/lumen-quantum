from scripts.check_canonical_urls import CANONICAL_OWNER, SLUG_PATTERN, non_canonical_references


def test_repository_uses_only_the_canonical_owner() -> None:
    assert non_canonical_references() == []


def test_pattern_flags_a_foreign_owner() -> None:
    sample = "see https://github.com/" + "someone-else" + "/lumen-quantum here"
    match = SLUG_PATTERN.search(sample)
    assert match is not None
    assert match.group(1) != CANONICAL_OWNER


def test_pattern_accepts_the_canonical_owner() -> None:
    sample = "https://github.com/" + CANONICAL_OWNER + "/lumen-quantum"
    match = SLUG_PATTERN.search(sample)
    assert match is not None
    assert match.group(1) == CANONICAL_OWNER
