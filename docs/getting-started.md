# Getting started

## Supported platforms

Python 3.11 and 3.12 are supported on Windows, Linux, and macOS. CPU-only
validation is the default. CUDA use is optional and outside the default CI path.

## Lightweight contributor setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run:

```bash
ruff format --check .
ruff check .
mypy
python scripts/validate_repository_configs.py
python scripts/repository_health.py
python scripts/check_markdown_links.py
python scripts/check_secrets.py
python -m pytest tests/test_repository_health.py -m unit
```

## ML setup

`requirements-dev.txt` adds the bounded CPU-capable ML stack and is a larger
installation:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -m "not slow and not gpu and not network"
```

Do not run dataset/model downloads or GPU/slow tests merely to validate a docs,
configuration, or maintenance change.

## Local smoke workflow

The historical smoke pipeline requires contributor-owned local text in ignored
`data/raw/`. Review the detailed legacy commands in the Git history and existing
topic docs, then run only the relevant tokenizer, data, and `--max-steps` smoke
commands. Never commit resulting artifacts.
