# Tokenizers

## Pilot lineage

The repository records two historically relevant 16,384-token pilot tokenizer
identities. `quantum-1.6-pilot` requires the tokenizer with SHA-256
`be99b72377f3cb2ce1c875103d0324a2001ee5543a49e7c8fabfc1e384b1b6f6` and rejects
the incompatible older pilot hash
`33017b41667f3ac30a60ee383f9018494b4c2c382ab2e83c7d0d219cd7c4c140`.

## Echelon tokenizer

The Echelon configuration targets SentencePiece BPE, 32,768 vocabulary entries,
byte fallback, identity normalization, and explicit special-token IDs. The
tracked validation report contains 23 round-trip cases with zero failures. The
tracked checksum record lists the locally produced model and vocabulary, but
those binaries are ignored and not licensed for redistribution here.

## Release requirements

A tokenizer release needs its training-data provenance, exact code/config
revision, SentencePiece version, vocabulary and normalization settings, special
tokens, artifact checksums, compatibility tests, known limitations, and an
explicit tokenizer license.
