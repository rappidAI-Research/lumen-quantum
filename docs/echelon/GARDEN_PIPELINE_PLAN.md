# rappidAI Quantum — quantum-1-echelon Garden pipeline

> Production-data preparation and Echelon training remain incomplete unless a
> retained manifest and run evidence say otherwise.

## Current direction: Garden v2 for Quantum 1 Echelon 1B

The old 8B Garden configuration is retained as historical 506M-era preflight
evidence. The current strategic path is defined under
[`configs/echelon/1b/`](../../configs/echelon/1b/) and targets a German-first,
multi-source Base corpus for the new approximately 1B-parameter model.

The fixed Base target is **40B training tokens**. A 50B extension is conditional
and may be used only after the 40B quality/budget gate protects the Chat-stage
budget and safety reserve.

## Planned 40B mix

| Domain | Planning share | Planning tokens |
|---|---:|---:|
| German HQ web | 65% | 26.0B |
| English educational content | 15% | 6.0B |
| Reference / knowledge | 8% | 3.2B |
| Mathematics | 7% | 2.8B |
| Code | 5% | 2.0B |

This is a planning mix, not an approved rights registry. The versioned
[`sources.yaml`](../../configs/echelon/1b/sources.yaml) must expose immutable
revisions, terms/review state and removal obligations before each source is
accepted for production.

## Garden v2 contract

Source registry → streaming/projection → language/quality filters →
PII/sensitive filtering → exact deduplication → documented source-specific
near-duplicate policy → benchmark decontamination → stable splits → frozen
Tokenizer → 4K sequence packing → 100M-token shards → hashes/manifests → private
recovery storage.

Rules:

- Do not reuse Quantum 1 or Quantum 1.6 training artifacts as hidden inputs.
- Do not infer a source license from public availability.
- Pin dataset revisions before production.
- Project away unneeded heavy columns early rather than retaining them through
  the whole pipeline.
- Preserve source/filter/split counts and restart evidence.
- Keep Validation and Test separate from training and tuning.
- Perform benchmark decontamination before the final production shards are
  declared ready.
- Hash every final shard and retain a machine-readable manifest.
- Do not start expensive GPU training before corpus QA and manifest review pass.

## Storage model

At `uint16`, 40B token IDs correspond to roughly 80 GB of raw token payload;
50B correspond to roughly 100 GB before metadata, indexes, temporary copies and
checkpoints. Raw sources should be streamed/projected where practical rather
than multiplied into unnecessary local copies.

Canonical shards and manifests are planned for private durable storage. Active
training copies may be cached on instance NVMe only after their checksums match
the canonical manifest.

## Historical Garden evidence

The earlier files under `configs/echelon/garden_*.yaml`, the pinned 8B production
configuration and the existing smoke report remain valid historical preflight
evidence. They must not be silently interpreted as the current 1B production
configuration or as proof that a final corpus exists.
