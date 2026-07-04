"""Tests fuer die Vorbereitung von quantum-1.6-pilot (Continued Pretraining).

Die reinen (torch-freien) Tests pruefen Konfiguration, Tokenizer-Kompatibilitaet,
neue Datenpfade und die Overlap-Vermeidung. Die torch-abhaengigen Tests
(weights-only Initialisierung, fehlende Checkpoints) werden uebersprungen, wenn
der ML-Stack nicht installiert ist, und laufen auf der GPU-Instanz.
"""

from pathlib import Path

import pytest

from scripts.exclude_known_documents import (
    build_exclusion_fingerprints,
    filter_new_documents,
    record_fingerprints,
)
from scripts.quantum_1_6_preflight import (
    EXPECTED_PARAMETER_COUNT,
    PreflightError,
    check_output_isolation,
    check_tokenizer_compatibility,
    load_yaml_config,
    run_preflight,
    validate_train_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = PROJECT_ROOT / "configs" / "quantum_1_6_pilot_train.yaml"
DATA_CONFIG = PROJECT_ROOT / "configs" / "quantum_1_6_pilot_data.yaml"


# --- Konfigurationsvalidierung -------------------------------------------------

def test_train_config_is_valid():
    config = load_yaml_config(TRAIN_CONFIG)
    notes = validate_train_config(config)
    assert config["project"]["model_name"] == "quantum-1.6-pilot"
    assert config["model"]["parameter_count_expected"] == EXPECTED_PARAMETER_COUNT
    # Effektive Batchgroesse 8 * 4 * 512 = 16.384 Tokens.
    assert any("16384" in note for note in notes)


def test_learning_rate_is_conservative():
    config = load_yaml_config(TRAIN_CONFIG)
    assert config["training"]["learning_rate"] == pytest.approx(1e-4)
    assert config["training"]["warmup_steps"] == 500
    assert config["training"]["max_grad_norm"] == 1.0
    assert config["training"]["lr_scheduler"] == "cosine"


def test_step_count_matches_token_budget():
    config = load_yaml_config(TRAIN_CONFIG)
    block = config["data"]["block_size"]
    eff_tokens = config["training"]["batch_size"] * config["training"]["gradient_accumulation_steps"] * block
    assert eff_tokens == 16384
    # 500 Mio. Tokens / 16.384 ~= 30.518 Schritte.
    assert config["training"]["max_steps"] == 30518
    assert abs(config["training"]["max_steps"] - round(500_000_000 / eff_tokens)) <= 1


def test_weights_only_flags_are_required():
    config = load_yaml_config(TRAIN_CONFIG)
    config["init"]["reset_optimizer"] = False
    with pytest.raises(PreflightError):
        validate_train_config(config)


def test_missing_init_section_is_rejected():
    config = load_yaml_config(TRAIN_CONFIG)
    del config["init"]
    with pytest.raises(PreflightError):
        validate_train_config(config)


# --- Output-Isolation ----------------------------------------------------------

def test_output_dir_must_be_under_quantum_1_6():
    config = load_yaml_config(TRAIN_CONFIG)
    config["training"]["output_dir"] = "models/quantum-1-base"
    with pytest.raises(PreflightError):
        validate_train_config(config)


def test_new_data_paths_do_not_overwrite_previous_dataset():
    train_config = load_yaml_config(TRAIN_CONFIG)
    data_config = load_yaml_config(DATA_CONFIG)
    # Sauber: darf nicht werfen.
    check_output_isolation(train_config, data_config)
    for path in data_config["paths"].values():
        assert path.startswith("data/quantum/quantum_1_6_pilot/")
    # Manipuliert: neuer Pfad zeigt auf alte Daten -> muss werfen.
    data_config["paths"]["cleaned_dir"] = "data/quantum/cleaned"
    with pytest.raises(PreflightError):
        check_output_isolation(train_config, data_config)


# --- Tokenizer-Kompatibilitaet -------------------------------------------------

def test_tokenizer_is_frozen_and_identical_to_base_model():
    config = load_yaml_config(TRAIN_CONFIG)
    report = check_tokenizer_compatibility(config)
    assert report["tokenizer_name"] == "quantum-1-pilot"
    assert report["identical_to_base_model"] is True
    assert report["vocab_size"] == 16384


def test_full_preflight_passes_on_repo():
    report = run_preflight(TRAIN_CONFIG, DATA_CONFIG)
    assert report["tokenizer"]["identical_to_base_model"] is True
    assert report["expected_parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert report["base_model"]["weights_file"] in {"model.safetensors", "pytorch_model.bin"}


# --- Overlap-Vermeidung --------------------------------------------------------

def test_record_fingerprints_uses_configured_fields():
    record = {"sha256": "abc", "id": "doc1", "text": "..."}
    fps = record_fingerprints(record, ["sha256", "id"])
    assert fps == {"sha256:abc", "id:doc1"}


def test_filter_removes_overlapping_documents():
    old_docs = [{"sha256": "h1", "id": "a"}, {"sha256": "h2", "id": "b"}]
    exclusion = set()
    for doc in old_docs:
        exclusion |= record_fingerprints(doc, ["sha256", "id"])

    new_docs = [
        {"sha256": "h1", "id": "a", "text": "duplikat"},   # exakte Ueberschneidung
        {"sha256": "h3", "id": "c", "text": "neu"},         # neu
        {"sha256": "hX", "id": "b", "text": "gleiche id"},  # gleiche id -> Ueberschneidung
    ]
    kept, removed = filter_new_documents(new_docs, exclusion, ["sha256", "id"])
    assert [doc["id"] for doc in kept] == ["c"]
    assert len(removed) == 2


def test_build_exclusion_fingerprints_reads_multiple_files(tmp_path):
    directory = tmp_path / "old"
    directory.mkdir()
    (directory / "train.jsonl").write_text(
        '{"sha256": "h1", "id": "a"}\n{"sha256": "h2", "id": "b"}\n', encoding="utf-8"
    )
    (directory / "validation.jsonl").write_text('{"sha256": "h3", "id": "c"}\n', encoding="utf-8")
    fingerprints, stats = build_exclusion_fingerprints(
        [directory], ["train.jsonl", "validation.jsonl", "missing.jsonl"], ["sha256", "id"]
    )
    assert "sha256:h1" in fingerprints
    assert "id:c" in fingerprints
    assert stats["documents_scanned"] == 3


# --- torch-abhaengige Tests (laufen auf der GPU-Instanz) -----------------------

def test_weights_only_initialization_loads_base_weights():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("sentencepiece")

    from scripts.inspect_model_size import (
        build_quantum_llama_config,
        build_quantum_model,
        count_parameters,
        load_quantum_tokenizer_info,
    )
    from scripts.train_quantum_continued import initialize_from_base_weights, verify_parameter_count

    config = load_yaml_config(TRAIN_CONFIG)
    base_model = Path(config["init"]["from_model"])
    if not (base_model / "model.safetensors").exists():
        pytest.skip("Basismodell models/quantum-1-base/final ist lokal nicht vorhanden.")

    tokenizer_info = load_quantum_tokenizer_info(config)
    model = build_quantum_model(build_quantum_llama_config(config, tokenizer_info))
    assert count_parameters(model) == EXPECTED_PARAMETER_COUNT
    assert verify_parameter_count(model) == EXPECTED_PARAMETER_COUNT

    report = initialize_from_base_weights(model, base_model)
    assert report["weights_changed"] is True
    assert report["embedding_sum_before_load"] != report["embedding_sum_after_load"]


def test_missing_checkpoint_resume_raises(tmp_path):
    pytest.importorskip("torch")
    from scripts.train_quantum_pilot import resolve_resume_checkpoint

    assert resolve_resume_checkpoint(tmp_path, None) is None
    with pytest.raises(FileNotFoundError):
        resolve_resume_checkpoint(tmp_path, "auto")
    with pytest.raises(FileNotFoundError):
        resolve_resume_checkpoint(tmp_path, str(tmp_path / "checkpoints" / "checkpoint-step-99999"))
