# Your first contribution

rappidAI Quantum is an early, sole-maintained research project. Small, precise
contributions are the most useful. Every change is reviewed by the maintainer.

## Set up in a few minutes (no GPU, no downloads)

```bash
git clone https://github.com/rappidAI-Research/lumen-quantum.git
cd lumen-quantum
python -m pip install -e ".[dev]"
python examples/repo_health_quickstart.py
```

The quickstart validates configs and runs the repository-health, secret, link,
and canonical-URL checks.

## Good first changes

- Fix a documentation error or an inconsistent command.
- Add a unit test for a configuration rule.
- Test that documented commands work across platforms.
- Check model-card consistency between [../model_cards/](../model_cards/) and the
  Hugging Face drafts in [hugging-face/](hugging-face/).
- Improve a small synthetic fixture or add a JSON-schema test.

See the concrete drafts in [issue-drafts/](issue-drafts/).

## Before you open a pull request

```bash
ruff format --check .
ruff check .
mypy
python scripts/validate_repository_configs.py
python scripts/repository_health.py
python scripts/check_canonical_urls.py
python scripts/check_markdown_links.py
python scripts/check_secrets.py
python scripts/build_run_manifest.py --validate-provenance reports/provenance/historical-run-provenance-status.json
python scripts/release_preflight.py --check
python -m pytest -m "unit and not slow and not gpu and not network"
```

Please read [../CONTRIBUTING.md](../CONTRIBUTING.md) and
[../CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md). Keep pull requests small and
focused, and describe what you changed and how you tested it.

## What not to do

- Do not commit model weights, tokenizer binaries, datasets, or large files.
- Do not add network or GPU requirements to the default test path.
- Do not describe pilot artifacts as "open weight"; their reuse licenses are
  unresolved (see [../MODEL_LICENSES.md](../MODEL_LICENSES.md)).
