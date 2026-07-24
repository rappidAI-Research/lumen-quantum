# Model and tokenizer license registry

Apache-2.0 covers only the source repository (code, configs, tests, original
docs). It does **not** cover the artifacts below. For every artifact whose reuse
terms are not published, the required statement is:

> No explicit reuse license published. Do not describe as open weight.

Fields per artifact: hosting repository, exact reviewed revision, rights holder,
source-code license, weight license, tokenizer license, redistribution,
modification, commercial use, publication status, verification status, and the
required maintainer action.

## `quantum-1-pilot` (model weights / GGUF)

| Field | Value |
|---|---|
| Hosting repository | `rappidAI/quantum-1-pilot` (Hugging Face) |
| Exact reviewed revision | `7daf415ef09fc131d7440af8514a93fd8cf3f2a1` |
| Rights holder | rappidAI (maintainer Jonas Désiré Cikemgil); legal entity Not verified |
| Source-code license | Apache-2.0 applies to this repo only, not the weights |
| Weight license | No explicit reuse license published |
| Tokenizer license | Unknown |
| Redistribution allowed | Unknown |
| Modification allowed | Unknown |
| Commercial use status | Unknown |
| Publication status | Experimental F16 GGUF published |
| Verification status | Publisher-reported |
| Required maintainer action | Choose and publish (or explicitly withhold) a weight license; do not describe as open weight |

## `quantum-1.6-pilot` (model weights / GGUF)

| Field | Value |
|---|---|
| Hosting repository | `rappidAI/quantum-1.6-pilot` (Hugging Face) |
| Exact reviewed revision | `507662c095b5ba6e14f24d3fc7f0a5e29d76b7f3` |
| Rights holder | rappidAI (maintainer Jonas Désiré Cikemgil); legal entity Not verified |
| Source-code license | Apache-2.0 applies to this repo only, not the weights |
| Weight license | No explicit reuse license published |
| Tokenizer license | Unknown |
| Redistribution allowed | Unknown |
| Modification allowed | Unknown |
| Commercial use status | Unknown |
| Publication status | Experimental F16 GGUF published; metrics publisher-reported |
| Verification status | Publisher-reported |
| Required maintainer action | Choose and publish (or explicitly withhold) a weight license; do not describe as open weight |

## `quantum-1-echelon` (model line)

| Field | Value |
|---|---|
| Hosting repository | None (no model artifact exists) |
| Exact reviewed revision | Code `f7eda1fb0ae153f0f9cc3477ead997cbdb462b39`; no model revision |
| Rights holder | rappidAI (maintainer Jonas Désiré Cikemgil) |
| Source-code license | Apache-2.0 for source only |
| Weight license | Not applicable yet (no weights) |
| Tokenizer license | Not selected |
| Redistribution allowed | Not applicable |
| Modification allowed | Not applicable |
| Commercial use status | Not applicable |
| Publication status | Architecture/tokenizer/data preflight only; Incomplete |
| Verification status | Not yet completed |
| Required maintainer action | Do not claim a trained model, checkpoint, or GGUF; choose terms before any future release |

## Pilot tokenizers

| Field | Value |
|---|---|
| Hosting repository | Referenced by code and pilot releases |
| Exact reviewed revision | Unknown (public artifact revision not confirmed) |
| Rights holder | rappidAI (maintainer Jonas Désiré Cikemgil) |
| Source-code license | Tokenizer configs in-repo are Apache-2.0; trained binaries are not |
| Weight license | Not applicable |
| Tokenizer license | No explicit reuse license published |
| Redistribution allowed | Unknown |
| Modification allowed | Unknown |
| Commercial use status | Unknown |
| Publication status | Referenced; explicit terms Not verified |
| Verification status | Not verified |
| Required maintainer action | Confirm public revision and publish an explicit tokenizer license |

## Echelon tokenizer artifacts

| Field | Value |
|---|---|
| Hosting repository | Binaries not tracked; checksums recorded in-repo |
| Exact reviewed revision | Code `f7eda1fb0ae153f0f9cc3477ead997cbdb462b39` |
| Rights holder | rappidAI (maintainer Jonas Désiré Cikemgil) |
| Source-code license | Configs Apache-2.0; binaries not covered |
| Weight license | Not applicable |
| Tokenizer license | No release license selected |
| Redistribution allowed | No (until a license is chosen) |
| Modification allowed | Unknown |
| Commercial use status | Unknown |
| Publication status | Checksums reported (`97dce887…`, `0fdfef7b…`, `36d3a745…`); binaries ignored |
| Verification status | Partial (checksums verified in tracked report) |
| Required maintainer action | Select a tokenizer license before any distribution |

## GGUF artifacts

| Field | Value |
|---|---|
| Hosting repository | `rappidAI/quantum-1-pilot`, `rappidAI/quantum-1.6-pilot` (Hugging Face) |
| Exact reviewed revision | See pilot rows above |
| Files / checksums | `quantum-1-base-v1.0.0-f16.gguf` `sha256 aeab97e5…` (98,990,560 B); `quantum-1.6-pilot-v1.6.0-f16.gguf` `sha256 6bda15fc…` (98,990,560 B) |
| Weight license | No explicit reuse license published |
| Redistribution allowed | Unknown |
| Modification allowed | Unknown |
| Commercial use status | Unknown |
| Publication status | Published F16 GGUF |
| Verification status | Checksums/sizes verified in model cards; terms Publisher-reported |
| Required maintainer action | Attach an explicit artifact license; do not describe as open weight |

The Apache-2.0 source license does not apply to any artifact above. Any derived
model must document its upstream model identifier, exact revision, license,
modifications, tokenizer, dataset terms, and compatibility obligations.
