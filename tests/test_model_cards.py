from scripts.check_model_cards import card_errors


def test_model_cards_are_consistent() -> None:
    assert card_errors() == []
