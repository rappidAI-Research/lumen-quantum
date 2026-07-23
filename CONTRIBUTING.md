# Contributing

Contributions are welcome. This is an experimental, sole-maintainer project, so
review or response times are not guaranteed.

## Setup

Use Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
pre-commit install
```

PowerShell activation is `.\.venv\Scripts\Activate.ps1`; the remaining Python
commands are the same.

For documentation and lightweight checks without the ML stack:

```bash
python -m pip install -e ".[dev]"
python scripts/validate_repository_configs.py
python scripts/repository_health.py
python -m pytest tests/test_repository_health.py -m unit
```

## Pull requests

- Open a focused issue or PR with observable acceptance criteria.
- Keep data, checkpoints, weights, credentials, personal information, and local
  paths out of Git.
- Mark tests that need `slow`, `gpu`, or `network` resources.
- Record data/model provenance and exact revisions for every new dependency.
- Run Ruff, MyPy, fast tests, configuration checks, link checks, and the secret
  scan described in `docs/getting-started.md`.
- Explain research claims as verified facts, publisher-reported results,
  configured targets, or incomplete work.

Contributions are accepted under Apache-2.0 unless clearly designated otherwise.
