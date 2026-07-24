from scripts.capture_environment import capture, installed_packages, key_versions, torch_details


def test_capture_has_core_sections() -> None:
    snapshot = capture()
    for key in (
        "python",
        "os",
        "platform",
        "cpu",
        "torch",
        "key_libraries",
        "installed_packages",
        "git",
    ):
        assert key in snapshot
    assert snapshot["python"]["version"]
    assert isinstance(snapshot["git"]["dirty"], bool)


def test_installed_packages_is_nonempty_mapping() -> None:
    packages = installed_packages()
    assert isinstance(packages, dict)
    assert packages  # pytest itself is installed


def test_key_versions_reports_all_names() -> None:
    versions = key_versions()
    assert set(versions) == {
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "sentencepiece",
        "tokenizers",
        "safetensors",
        "numpy",
        "PyYAML",
    }


def test_torch_details_shape() -> None:
    details = torch_details()
    assert "available" in details
    assert isinstance(details["available"], bool)
