# Echelon compute plan

Reviewed: 2026-09-18. Evidence baseline:
`f79c395c1da82827a63706ea9b55c913c073bd91`.

**Status: planning, not an executed cloud deployment or training run.** No AWS
credits, account eligibility, GPU quota, capacity, training throughput, or
completed Echelon checkpoint is asserted here. This document does not authorize
data preparation, downloads, cloud spending, or training.

## Purpose and next milestone

rappidAI is a self-funded, founder-led, early-stage AI research and development
initiative based in Berlin. Its current model-development priority is to move
`quantum-1-echelon` from documented preflight and smoke tests to a reviewed
production corpus, then a separately authorized base-training and evaluation
run. Compute and storage are execution constraints; acquiring them alone does
not resolve data rights, missing training configuration, or release gates.

Quantum is the model-development project; Lumen is its historical name. The
published `quantum-1-pilot` and `quantum-1.6-pilot` are experimental pilots, not
evidence of completed Echelon work. Ghost and Replay provide supporting
engineering context, but their workloads are outside this Echelon budget.

## Verified starting point

| Stage | Retained evidence | Boundary |
|---|---|---|
| Architecture | [Configuration](../configs/echelon/quantum-1-echelon-base.yaml) and [preflight](../reports/quantum-1-echelon/quantum-1-echelon-base-preflight.json) | Computed parameter count, not trained weights |
| Tokenizer | [Configuration](../configs/echelon/tokenizer.yaml), [23 passing round-trip cases](../reports/quantum-1-echelon/tokenizer_validation.json), [quality sample](../reports/quantum-1-echelon/tokenizer_quality.json), [checksums](../reports/quantum-1-echelon/tokenizer_SHA256SUMS.txt) | Reports are public; trained binaries are not tracked; not model evaluation |
| Garden smoke | [Phase 3 report](../reports/quantum-1-echelon/garden_phase3_report.md): 5,001 documents seen, 1,559 accepted, 1,380,886 tokens | Smoke, integrity, interruption, resume and shutdown evidence only |
| Production data | [Pinned production configuration](../configs/echelon/garden_production.yaml) | No completed production manifest or final corpus |
| Training, evaluation, release | [Model card](../model_cards/quantum-1-echelon.md) and [open production issue](https://github.com/rappidAI-Research/lumen-quantum/issues/6) | No trained checkpoint, raw model evaluation, GGUF or release |

## Model and data scale

The base model is a Llama-style causal decoder with **506,333,440 parameters**
(500M target; accepted range 475–525M), 26 layers, hidden size 1,280,
intermediate size 3,584, 20 attention heads, 5 KV heads, tied embeddings,
32,768 vocabulary, 2,048-token context and BF16 precision target.

The [Garden plan](echelon/GARDEN_PIPELINE_PLAN.md) distinguishes a minimum 5B
training-token target from a preferred 8–10B. The executable production config
currently requests 8,000,000,000 training tokens, 10,000,000 validation tokens,
10,000,000 test tokens and 100,000,000 tokens per shard. Targets are not
completed totals; stop/overshoot behavior and actual split counts must appear
in the final manifest.

Source: `epfml/FineWeb2-HQ`, configuration `deu_Latn`, train split, streaming,
revision `c0c06e94fd3a44ae9e802b2b0fc533817601eb5e`, seed `20260718`.
ODC-By 1.0 collection terms do not extinguish source-page rights or personal-data
and removal obligations; see [DATA_SOURCES](../DATA_SOURCES.md).

The early Garden planning config requests near-duplicate filtering. The later
production config explicitly disables an additional in-memory near-duplicate
index because it relies on upstream per-language MinHash deduplication and
filters cluster metadata. Record this actual behavior; do not claim a second
production-wide deduplication pass. Review its adequacy before execution.

## Workload stages and acceptance gates

1. **Production data preparation, CPU first:** review data rights, removal
   handling, pinned inputs, filters and budget. Stream and filter the source;
   retain acceptance/rejection counts, state checkpoints and restart evidence.
2. **Tokenization and sharding:** verify the actual tokenizer against the
   committed checksum; produce `uint16` shards with EOS boundaries, deterministic
   splits, checksums and final manifest. Review corpus quality and leakage before
   authorizing training. Keep expensive GPU instances off during these stages.
3. **Training readiness:** commit an Echelon-specific training recipe covering
   the Garden shard loader, tokenizer, optimizer, schedule, microbatch,
   accumulation, sequence length, seed, stopping rule and checkpoint retention.
   The architecture config is not a complete training recipe. Pilot workflows
   must not silently substitute their data, tokenizer or continued-pretraining
   initialization for Echelon.
4. **Bounded hardware preflight, separately authorized:** test at the actual
   2,048-token context with forward, backward and optimizer steps, then verify
   checkpoint save/reload and deterministic resume behavior where supported.
   Record peak allocated/reserved memory, tokens/second and environment. The
   existing GPU smoke uses a batch of one at 128 tokens and no optimizer step;
   it cannot establish production training fit or cost.
5. **Base training and checkpoints:** train only after data and hardware gates
   pass. Preserve optimizer, scheduler, RNG and data position; verify restart
   before relying on interruption-tolerant capacity.
6. **Evaluation:** retain raw held-out loss/perplexity and fixed-prompt outputs
   with exact settings. Keep the test split separate from tuning. Report quality,
   safety and generalization limits without inferring benchmark leadership.
7. **Export and publication:** verify the external llama.cpp revision, export
   compatibility and artifact hashes; publish a final manifest and evidence.
   Model/tokenizer licenses and release approval are separate gates. Echelon
   Chat remains a later stage within this line.

## GPU memory analysis

For `N = 506,333,440`, the existing preflight models the following static state:

| Allocation | Calculation | GiB |
|---|---|---:|
| BF16 weights | `2 × N / 2^30` | 0.943 |
| BF16 gradients | `2 × N / 2^30` | 0.943 |
| FP32 master weights | `4 × N / 2^30` | 1.886 |
| FP32 AdamW moments | `8 × N / 2^30` | 3.772 |
| Static total | `16 × N / 2^30` | 7.545 |

This is a representation assumption, not a measurement of a particular
PyTorch/optimizer implementation. Verify actual parameter, gradient, master
copy and optimizer-state dtypes. Autocast alone does not guarantee BF16 storage.

Activations, attention intermediates, logits, temporary optimizer tensors,
CUDA workspaces and fragmentation are additional. At batch one and full
context, a single BF16 `[2048, 32768]` logits tensor is 0.125 GiB. Naively
materializing attention scores for 20 heads costs 0.156 GiB per layer in BF16
before backward storage; an efficient attention kernel changes that behavior.
These examples are not an additive total-memory prediction.

A full 24 GB GPU is a plausible candidate with a small microbatch, gradient
accumulation and, if needed, activation checkpointing/efficient attention.
That fit is **unmeasured**. A 48 GB GPU provides more headroom. One GPU is not
ruled out by the static state; multiple GPUs are not a demonstrated requirement.
Distributed training adds implementation, communication and checkpoint costs
and must be justified by measured throughput or fit. Do not equate GPU model
names or advertised memory with proven workload performance.

## Storage and checkpoint budget

[garden_pipeline.yaml](../configs/echelon/garden_pipeline.yaml) provides a
300 GB working budget: 40 GB raw cache, 80 GB cleaned data, 30 GB tokenized data,
120 GB training artifacts and 30 GB safety buffer. These are planning caps;
the production script does not enforce this entire budget automatically.

`8,020,000,000 × 2` bytes is **16.04 GB (14.94 GiB)** of token payload at the
configured target; each full 100M-token shard is 200 MB. This excludes metadata,
partial shards, caches, intermediate representations and copies. Streaming
can reduce raw-data retention but cannot eliminate checkpoint and recovery
storage. Preserve input provenance before deleting intermediates.

BF16 weights alone are about 1.01 GB. A resumable checkpoint additionally needs
optimizer state and potentially FP32 model/master copies; its actual serialized
size must be measured. Gradients are not assumed to be checkpointed. Account
for both the previous valid checkpoint and the temporary replacement during
atomic writes. A proposed initial policy is two rotating recovery checkpoints
plus one best/final checkpoint, subject to the 120 GB artifacts cap. Retention
must be enforced explicitly before a run, not assumed from this document.

Use durable EBS for active state and private S3 for verified recovery copies.
Instance-store NVMe is scratch space, not the only copy of evidence. Budget EBS,
S3, snapshots, requests and transfer independently; a 300 GB working volume is
not a 300 GB total billed-storage ceiling.

## AWS-compatible options

Official hardware pages checked on 2026-09-18; region availability, quotas,
capacity, usable memory and prices must be checked again before allocation.
The default region to price is Europe (Frankfurt), `eu-central-1`; this is a
planning choice, not a deployed region or a data-residency guarantee.

| Option | Published hardware | Role in this plan |
|---|---|---|
| [EC2 G6](https://aws.amazon.com/ec2/instance-types/g6/) | NVIDIA L4, 24 GB per full GPU | First cost/fit comparison for bounded preflight and small-model training; avoid fractional GPU variants for this baseline |
| [EC2 G6e](https://aws.amazon.com/ec2/instance-types/g6e/) | NVIDIA L40S, 48 GB per GPU | Single-GPU headroom alternative; AWS explicitly describes smaller-model training as a use case |
| [EC2 G7e](https://aws.amazon.com/ec2/instance-types/g7e/) | RTX PRO 6000 Blackwell Server Edition, 96 GB per GPU | Larger-memory alternative only if fit or measured cost per token justifies it |
| [EC2 P5](https://aws.amazon.com/ec2/instance-types/p5/) | H100 family; P5e/P5en use H200 | Higher-end comparison, not an Echelon requirement |

Choose host RAM and vCPUs for streaming/tokenization and data-loader behavior,
not only GPU memory. Compare total cost per successfully trained token rather
than hourly rate alone. Do not reserve capacity, buy commitments or request
paid support as part of readiness work.

## Credit utilization and costing

No credits have been assumed granted. AWS currently describes Founders as
starting at USD 1,000, with selected participants potentially qualifying for up
to USD 5,000. Approval and applicable services/expiry depend on the actual offer.
See the [official program page](https://aws.amazon.com/startups/credits/) and
[promotional credit terms](https://aws.amazon.com/awscredits/).

Proposed allocation of an eventual approved, eligible-service budget `B`:

| Work | Planning ceiling |
|---|---:|
| GPU preflight and base training | 60% of B |
| CPU preparation and tokenization | 15% of B |
| Active durable storage | 10% of B |
| Recovery copies and snapshots | 5% of B |
| Evaluation/export and contingency | 10% of B |

These are initial allocation limits, not measured costs or an assertion that
the complete 8B-token run fits. Start with the bounded readiness milestone;
stop at its cap if the measured full-run estimate exceeds available funding.

For measured aggregate throughput `q` tokens/second and target `T`, compute
`training_hours = T / q / 3600`. Add measured startup, evaluation, checkpoint,
restart and export time. Then price GPU/CPU hours, GB-month storage, snapshot
retention, requests, transfer, applicable IPv4/networking and uncovered charges.
No throughput or dollar-per-hour figure is invented here.

Before authorizing a workload, save a dated regional quote from the
[AWS Pricing Calculator](https://calculator.aws/) and current
[EC2](https://aws.amazon.com/ec2/pricing/on-demand/),
[EBS](https://aws.amazon.com/ebs/pricing/) and
[S3](https://aws.amazon.com/s3/pricing/) rates. Record region, instance, OS,
tenancy, purchase option, currency, taxes, storage units, rate date and credit
exclusions. A fixed hourly quote and a completion-cost estimate remain open
until capacity, the final recipe and throughput are known. Spot is optional
only after interruption recovery is demonstrated; its discounts are not a
guaranteed budget assumption.

## Reproducibility requirements

Retain the code commit; exact config and hash; seeds; dataset ID, configuration
and immutable revision; tokenizer binaries/checksums under appropriate access;
environment/dependency record; hardware, drivers, kernels and precision;
filter/split/token counts; shard checksums; optimizer/scheduler/RNG/data position;
checkpoint hashes and resume evidence; raw evaluation outputs; export/tool
revision; final manifest, completion status and artifact checksums. Use the
existing [run-manifest schema](../schemas/run-manifest.schema.json),
[environment capture](environment-capture.md) and
[reproducibility guide](reproducibility.md). Unknown historical fields stay
unknown; a new revision must never be substituted as historical provenance.

## Planned cost and security controls

- Set account/workload budgets and alerts before provisioning. Alerts can lag
  and are not a hard spending cap; also enforce job wall-time limits, shutdown
  on completion/failure and an independently tested idle-instance watchdog.
- Keep GPUs off during CPU preparation. Inventory instances, volumes, snapshots
  and buckets after each stage; stopping an instance does not remove storage
  charges. Apply bounded retention and lifecycle cleanup only after verifying
  retained manifests and recovery copies.
- Use temporary role credentials, least-privilege IAM and MFA for human access.
  Never put credentials, account identifiers or private billing data in Git.
- Keep buckets private with public access blocked and encryption enabled; scope
  access to the workload. Do not expose training/control endpoints publicly.
- Resume only from trusted, provenance-checked checkpoints. Existing optimizer
  resume state may contain pickle data; hashes do not make an untrusted producer
  safe. Preserve the boundaries in [SECURITY](../SECURITY.md).
- Implement and test these AWS controls before execution. No Terraform,
  account-level policy, budget alarm or AWS access deployment is claimed here.

## Application boundary

This plan supports an honest description of future infrastructure use. It is
not an AWS affiliation, funding announcement or proof of eligibility. The
initiative's public legal status remains private individual / independent
initiative. Verify account plan, founding date, prior credits and the current
application requirements separately; see
[the application readiness record](applications/aws-activate-readiness.md).
