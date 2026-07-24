from scripts import release_preflight


def test_required_and_approval_files_exist() -> None:
    assert release_preflight.required_file_errors() == []


def test_version_is_consistent_across_metadata() -> None:
    assert release_preflight.version_errors() == []


def test_no_unsupported_public_claims() -> None:
    assert release_preflight.banned_claim_errors() == []


def test_project_version_is_a_prerelease() -> None:
    assert release_preflight.VERSION_PATTERN.match(release_preflight.project_version())


def test_collect_reports_named_checks_and_is_clean() -> None:
    results = release_preflight.collect(check_worktree=False)
    for name in (
        "required release and approval files",
        "version consistency",
        "unsupported public claims",
        "canonical repository URLs",
    ):
        assert name in results
    substantive = {name: problems for name, problems in results.items() if problems}
    assert substantive == {}, substantive
