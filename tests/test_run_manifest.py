import json

from scripts.build_run_manifest import (
    ROOT,
    build_manifest,
    load_schema,
    validate_against_schema,
    validate_provenance_entry,
)

PROVENANCE = ROOT / "reports" / "provenance" / "historical-run-provenance-status.json"
REQUIRED = {
    "schema_version",
    "generated_at",
    "code",
    "configuration_files",
    "python",
    "platform",
    "dependencies",
    "seeds",
    "dataset",
    "model",
    "tokenizer",
    "timing",
    "completion_status",
    "error_status",
    "verification_status",
}


def test_schema_declares_required_fields() -> None:
    assert load_schema()["required"]


def test_build_manifest_defaults_and_schema() -> None:
    manifest = build_manifest([ROOT / "configs" / "smoke_5m.yaml"])
    assert REQUIRED.issubset(manifest)
    assert manifest["completion_status"] == "metadata-only"
    assert manifest["verification_status"] == "unknown"
    assert manifest["seeds"] == 42
    assert manifest["configuration_files"][0]["sha256"]
    assert validate_against_schema(manifest) == []


def test_build_manifest_does_not_guess_unknowns() -> None:
    manifest = build_manifest([ROOT / "configs" / "smoke_5m.yaml"])
    assert manifest["model"]["checksum"] is None
    assert manifest["tokenizer"]["checksum"] is None
    assert manifest["error_status"] is None


def test_provenance_registry_is_valid() -> None:
    entries = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert isinstance(entries, list)
    assert entries
    for entry in entries:
        assert validate_provenance_entry(entry) == []
