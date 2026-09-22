# Quantum 1 Echelon 1B — tokenizer A/B gate

Status: tooling and fixed evaluation fixture are versioned; the production
training corpus has not been assembled and no candidate is frozen.

## Goal

Choose between the 32K/21-layer and 48K/20-layer approximately-1B candidates
without letting different tokenizer training data bias the comparison.

Both SentencePiece candidates must therefore train from **the exact same local
corpus bytes**. Their manifests record the corpus SHA-256, normalized config
SHA-256, SentencePiece version, requested/actual vocabulary, artifact hashes
and special-token IDs.

## Production corpus requirement

The target is a reproducible, representative tokenizer sample of roughly 100M
tokens worth of text, assembled only from source records that have passed the
Garden v2 provenance/terms gate. It should cover the same important domains as
the planned Base mix rather than using German web text alone.

Do not define "100M" by one candidate's tokenizer count and then give the other
candidate different text. Assemble one fixed corpus, freeze its SHA-256, then
train both candidates from that file.

## Commands

Once the shared corpus exists:

```bash
python scripts/echelon_tokenizer_ab.py train \
  --config configs/echelon/1b/tokenizer-32k.yaml \
  --corpus /path/to/shared-tokenizer-corpus.txt

python scripts/echelon_tokenizer_ab.py train \
  --config configs/echelon/1b/tokenizer-48k.yaml \
  --corpus /path/to/shared-tokenizer-corpus.txt
```

Then evaluate both on the same fixed JSONL evaluation corpus and pass each
candidate's generated `tokenizer-manifest.json` so the report retains the
training-corpus identity.

The tracked smoke/evidence fixture is
`data/evals/echelon_tokenizer_ab_v1.jsonl`; production review may add a larger
versioned evaluation set without silently replacing v1 evidence.

## Metrics

The report records, overall and per domain:

- bytes/token;
- characters/token;
- tokens/word;
- byte-fallback token rate;
- exact round-trip failures.

The comparison refuses candidates trained on different corpus hashes or
evaluated on different evaluation-corpus hashes.

## Decision rule

There is intentionally no automatic winner. **Prefer 32K** because it leaves
more of the fixed 1B parameter budget for transformer depth unless 48K shows a
material and repeatable efficiency/coverage advantage without German, English,
math, code or round-trip regressions.

The final decision must update the live STATUS record and freeze exactly one
architecture config plus one tokenizer manifest/checksum before Garden v2
production tokenization.
