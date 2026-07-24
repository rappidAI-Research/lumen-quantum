# Datasets

## FineWeb2-HQ use

The active configurations reference `epfml/FineWeb2-HQ`, usually `deu_Latn`.
The publisher states ODC-By 1.0 and Common Crawl Terms of Use. The collection is
web-derived, can include personal or harmful content, and does not erase rights
or removal obligations associated with source pages.

The Echelon production configuration pins
`c0c06e94fd3a44ae9e802b2b0fc533817601eb5e`. Older pilot configurations with
`revision: main` are reproducibility gaps and must be pinned before another run.

## Pipeline

The code streams or reads records, validates metadata, filters language and
quality, removes exact overlaps, assigns stable hash buckets, tokenizes accepted
documents, writes shards, and records manifests/checkpoints. Near-duplicate,
copyright, privacy, toxicity, and factual-quality filtering remain incomplete.

## Safe operation

- Do not commit raw, cleaned, or tokenized data.
- Do not log full sensitive documents.
- Record removal/opt-out handling before public data or model release.
- Treat URL, author, and metadata fields as potentially personal.
- Review `DATA_SOURCES.md` and the publisher's current dataset card before a run.
