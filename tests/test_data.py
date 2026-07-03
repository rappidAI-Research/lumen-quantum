from scripts.prepare_data import deduplicate_units, split_units


def test_deduplicate_removes_exact_duplicates_and_keeps_order():
    assert deduplicate_units(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_three_way_split_is_disjoint_and_covers_all_units():
    units = [f"unit-{i}" for i in range(10)]

    train, validation, test, strategy = split_units(
        units, validation_ratio=0.2, test_ratio=0.2, seed=42
    )

    assert strategy == "shuffled_unit_split"
    # Kein Verlust, keine Duplikate ueber die Splits hinweg.
    assert sorted(train + validation + test) == sorted(units)
    assert len(train) >= 1 and len(validation) >= 1 and len(test) >= 1
    # Kein Datenleck: Splits sind paarweise disjunkt.
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)


def test_split_is_deterministic_for_same_seed():
    units = [f"unit-{i}" for i in range(20)]

    first = split_units(units, 0.1, 0.1, seed=7)
    second = split_units(units, 0.1, 0.1, seed=7)

    assert first == second


def test_split_keeps_training_units_on_tiny_dataset():
    # Drei Einheiten wie im README-Beispiel: je 1x train/val/test.
    units = ["a", "b", "c"]

    train, validation, test, _ = split_units(units, 0.1, 0.1, seed=1)

    assert len(train) >= 1
    assert len(validation) >= 1
    assert sorted(train + validation + test) == ["a", "b", "c"]
