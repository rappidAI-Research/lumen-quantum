# AWS Activate application readiness

Reviewed 2026-09-18. This is a preparation record, not an application submission,
approval, credit award or legal-entity determination.

## Project facts

rappidAI is a self-funded, founder-led, early-stage AI research and development
initiative based in Berlin. Founder: Jonas Désiré Cikemgil. Project contact:
`cikemgil@rappidai-research.com`. Public legal status: private individual /
independent initiative; no incorporation, investors, customers or revenue is
asserted. Self-funding is a maintainer-provided statement, not independently
audited financial evidence.

- [Website](https://www.rappidai-research.com) and
  [its existing repository](https://github.com/desirecikemgil/rappidai-research.com).
- [Quantum](https://github.com/rappidAI-Research/lumen-quantum): experimental
  model development, with [pilot and Echelon status](../../README.md).
- [Ghost](https://github.com/rappidAI-Research/rappid-ghost): v0.2.0 released;
  v0.3 development is separate from stable release capabilities.
- [Replay](https://github.com/rappidAI-Research/rappid-replay): experimental
  execution recording and reproducibility infrastructure.
- [Source-license approval](../maintainer-source-license-approval.md) is
  recorded; [artifact terms](../../MODEL_LICENSES.md) remain separate. Published
  pilots are not open weight.

The concrete infrastructure use case is Echelon production-data preparation,
tokenization/sharding, a bounded training preflight, base training and evaluation,
subject to separate acceptance gates. See [compute plan](../compute-plan.md).
No production corpus, Echelon checkpoint or capability evaluation is claimed.

## Current AWS requirements and limits

The [official credits page](https://aws.amazon.com/startups/credits/) describes
Founders for self-funded startups, starting at USD 1,000, with selected
participants potentially qualifying for up to USD 5,000. General criteria
include being founded within ten years, pre-Series B, an AWS Paid Tier account,
and the applicable prior-credit conditions. The
[application guide](https://aws.amazon.com/aws-startups/learn/applying-for-aws-activate-credits-a-step-by-step-guide/)
specifies a functioning website and a business email matching the startup domain;
Founders is intended for applicants new to Activate credits. Use the current
application's exact requirements if offer wording changes.

The [AWS FAQ](https://startups.aws.com/faq) confirms that free account plans are
ineligible for Activate promotional credits. The reviewed public pages do not
establish whether this particular independent initiative will be accepted or
resolve every legal-form field in the application. Do not infer approval from
website polish, technical documentation or the absence of an explicit
incorporation requirement on those pages.

## Owner checks before submission

- Confirm the actual founding date; do not derive it from the first Git commit.
- Confirm prior AWS promotional-credit history and the relevant Founders rules.
- Confirm the AWS account and Builder ID association, business-domain email
  access, account plan and credit offer. Account identifiers stay private.
- If a Paid Tier upgrade is needed, the owner must review its billing
  implications and perform or explicitly authorize it separately.
- Answer legal-form questions truthfully. If the form cannot represent the
  initiative's status, obtain clarification from AWS rather than inventing a
  company registration or selecting an inaccurate entity type.
- Review data rights/removal handling, compute quote and execution gates before
  any workload. Credits are not a spending cap and may exclude charges.
- Submit the application only after reviewing every field; this repository pass
  neither submits it nor accepts program terms on the owner's behalf.

## Evidence gaps retained

Open issues cover [pilot artifact notices](https://github.com/rappidAI-Research/lumen-quantum/issues/2),
[historical data revisions](https://github.com/rappidAI-Research/lumen-quantum/issues/3),
[final pilot manifests/evaluations](https://github.com/rappidAI-Research/lumen-quantum/issues/4),
[CPU measurements](https://github.com/rappidAI-Research/lumen-quantum/issues/5)
and [Echelon production data](https://github.com/rappidAI-Research/lumen-quantum/issues/6).
Planning documentation does not close those evidence gaps. No new model release,
benchmark result, cloud deployment or production-readiness claim follows from
this preparation work.
