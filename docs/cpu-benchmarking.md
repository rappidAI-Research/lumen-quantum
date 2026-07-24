# CPU inference benchmarking

`scripts/benchmark_cpu_inference.py` measures latency and throughput for a model
directory that you supply explicitly. It never downloads a model, never contacts
the network, and never publishes results. No benchmark numbers are claimed in
this repository until the tool is run against a real local artifact and the
record is reviewed.

## Dry run (no model, no ML dependencies)

```bash
python scripts/benchmark_cpu_inference.py --dry-run
```

Runs the full measurement path with a deterministic mock generator. It verifies
the harness only; the reported speed is meaningless as a model measurement and
is labeled `dry_run: true`.

## Benchmarking a real local model

```bash
python -m pip install -e ".[ml]"
python scripts/benchmark_cpu_inference.py \
  --model-path /path/to/local/model \
  --prompts data/evals/cpu_benchmark_prompts.jsonl \
  --output reports/cpu/example-result.json \
  --threads 4 --warmup-runs 2 --measurement-runs 5
```

The model directory must already exist locally; it is loaded with
`local_files_only=True` and nothing is fetched.

## What the record captures

Model path and checksum, tokenizer checksum, prompt-file path and checksum, code
commit and dirty status, OS, CPU, RAM (where portable), requested threads,
Python and library versions, generation settings, warm-up and measurement runs,
per-run latency, latency mean/median, tokens-per-second mean/median, and process
peak memory where it can be read.

## Known limitations

- Latency and throughput depend on hardware, thread count, and build flags.
- Peak-memory units differ by platform and are reported verbatim.
- The small synthetic prompt set is not a task benchmark; results are
  operational measurements, not capability or comparison claims.

See [reproducibility.md](reproducibility.md), [environment-capture.md](environment-capture.md),
and the prompt fixture at
[../data/evals/cpu_benchmark_prompts.jsonl](../data/evals/cpu_benchmark_prompts.jsonl).
