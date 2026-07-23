# Security model

## Assets

Credentials, private data, source-corpus content, tokenizer/model artifacts,
checkpoints, evaluation outputs, contributor systems, and release integrity are
protected assets.

## Threats

- malicious pickle/checkpoint execution;
- compromised Python dependency or GitHub Action;
- unverified llama.cpp source/binary execution;
- dataset prompt injection, personal data, malware text, and harmful content;
- secret or local-path leakage through configs/logs;
- model or tokenizer substitution and checksum drift;
- denial of service through large files, contexts, or decompression; and
- overstated research claims that bypass release review.

## Controls

Actions are SHA-pinned with read-only permissions. Default CI disables GPU and
network access for tests and excludes marked slow/network/GPU tests. Repository
health rejects unregistered Gitlinks and tracked model/data binaries. Config,
local-link, and high-confidence secret checks run in CI. Release artifacts need
checksums and immutable provenance.

## Residual risk

The secret scan is intentionally narrow, dependency resolution is not fully
locked, no independent security team exists, datasets are not comprehensively
audited, and ML artifacts are not sandboxed. Follow `SECURITY.md` for private
reports.
