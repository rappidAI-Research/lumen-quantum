# Third-party notices

This file records external components referenced by the repository. It is not a
substitute for reading their complete terms.

## Python dependencies

Dependencies are installed from their upstream distributions and are not
vendored here. Their licenses remain their own. The authoritative dependency
ranges are in `pyproject.toml`; review the resolved environment before a release.

## FineWeb2-HQ

The data configurations reference `epfml/FineWeb2-HQ`, primarily the
`deu_Latn` configuration. Its dataset card states ODC-By 1.0 and says use is
also subject to Common Crawl's terms. Source documents may retain separate
rights, contain personal information, or require removal. No dataset rows are
tracked in this repository. See `DATA_SOURCES.md`.

## llama.cpp

GGUF conversion and optional local inference use an external checkout of
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp). The previously
recorded, tested revision is
`d4cff114c0084f1fbc9b4c62717eca8fb2ae494a`, licensed upstream under MIT at that
revision. The code is not vendored or sublicensed by this repository.

## Contributor Covenant

`CODE_OF_CONDUCT.md` is based on Contributor Covenant 2.1, available under
Creative Commons Attribution 4.0.
