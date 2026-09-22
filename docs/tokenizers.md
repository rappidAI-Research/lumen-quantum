# Tokenizers

## Pilot lineage

The repository records two historically relevant 16,384-token pilot tokenizer
identities. `quantum-1.6-pilot` requires the tokenizer with SHA-256
`be99b72377f3cb2ce1c875103d0324a2001ee5543a49e7c8fabfc1e384b1b6f6` and rejects
the incompatible older pilot hash
`33017b41667f3ac30a60ee383f9018494b4c2c382ab2e83c7d0d219cd7c4c140`.

## Echelon tokenizer

The retained historical Echelon tokenizer targets SentencePiece BPE with 32,768
entries and has tracked round-trip/checksum evidence. For the strategic 1B path,
32K and 48K candidates are explicitly versioned under `configs/echelon/1b/`.
Neither is frozen or claimed trained. The production tokenizer is selected only
after German/English/Code/Math fertility, byte fallback, round-trip behavior and
checksums are compared on a representative sample of the planned final mix.

## Release requirements

A tokenizer release needs its training-data provenance, exact code/config
revision, SentencePiece version, vocabulary and normalization settings, special
tokens, artifact checksums, compatibility tests, known limitations, and an
explicit tokenizer license.
