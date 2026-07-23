# GGUF export

llama.cpp is an external tool. A broken Gitlink previously pointed at commit
`d4cff114c0084f1fbc9b4c62717eca8fb2ae494a` without `.gitmodules`; it was removed
instead of converted to a submodule.

## Prepare the external dependency

```bash
git clone https://github.com/ggml-org/llama.cpp.git /path/to/llama.cpp
git -C /path/to/llama.cpp checkout d4cff114c0084f1fbc9b4c62717eca8fb2ae494a
git -C /path/to/llama.cpp rev-parse HEAD
```

Review the upstream MIT license and source before executing it. A newer revision
requires compatibility validation and a recorded reason; do not silently move
the tested baseline.

## Export

```bash
python scripts/export_gguf.py \
  --model-dir models/smoke/final \
  --output-file models/smoke/quantum-smoke-f16.gguf \
  --llama-cpp-dir /path/to/llama.cpp
```

PowerShell uses the same arguments with a Windows path, for example
`--llama-cpp-dir C:\path\to\llama.cpp`.

The wrapper validates local config, weights, tokenizer files and special-token
compatibility, stages a conversion directory, and invokes the upstream Python
converter without a shell. Generated GGUF files remain ignored and need their
own license, manifest, size, and checksum before distribution.

For the 1.6 diagnosis `gguf` or `all` modes, pass the same explicit
`--llama-cpp-dir` option.
