# Environment capture

`scripts/capture_environment.py` records a resolved runtime snapshot so each
experiment can be reproduced against a known environment. It complements the
bounded version ranges in `pyproject.toml`: the ranges describe what is allowed,
the snapshot records what was actually installed.

```bash
python scripts/capture_environment.py --output reports/environments/example.json
```

## Captured fields

- Python version and implementation.
- Operating system and release, platform string.
- CPU processor and machine.
- PyTorch availability, version, CUDA version, CUDA availability, and GPU names
  (only when PyTorch is importable).
- Key ML library versions (`torch`, `transformers`, `datasets`, `accelerate`,
  `sentencepiece`, `tokenizers`, `safetensors`, `numpy`, `PyYAML`).
- Every installed package with its exact version.
- Git commit and working-tree dirty status.

The output is JSON. No value is invented; missing tools are reported as `null`.
Capture the snapshot next to any run whose results you intend to cite.
