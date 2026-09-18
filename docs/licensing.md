# Licensing boundaries

## Source code and documentation

Original repository source code, configurations, tests, and project
documentation are licensed under Apache License 2.0, which is now present on
`main`. Git history attributes all tracked original material to the maintainer
account, and the audit found no vendored copied source. The maintainer separately
confirmed source/documentation ownership and approved Apache-2.0 on 2026-07-24.
This is a reference to the existing human decision, not a new automated legal
determination. See
[maintainer-source-license-approval.md](maintainer-source-license-approval.md).

## Excluded material

The root license does not silently license:

- model weights or GGUF files;
- tokenizer models, vocabularies, or derived tokenizer artifacts;
- raw, cleaned, or tokenized dataset content;
- third-party source, executables, or external repositories;
- reports that reproduce third-party text; or
- project names, logos, trade dress, or trademarks.

Each released artifact needs an explicit, adjacent license and provenance
record. Public download access is not a reuse license.

## Recorded approval

The dated human decision is retained in the linked approval record. Its scope
covers the following confirmations; new contributions still need compatible
rights:

1. they own or have permission to license all original tracked code and prose;
2. no employer, client, collaborator, or contract holds conflicting rights;
3. no copied snippet lacks its source, notice, or compatible license;
4. generated reports contain no redistributable third-party content beyond what
   the relevant terms permit; and
5. Apache-2.0 is the intended license for future inbound contributions.

The maintainer also recorded a deliberate all-rights-reserved decision for pilot
weights, GGUFs and tokenizers on 2026-07-24. No reuse license is granted. Explicit
notices in the hosting repositories and artifact revision verification remain
publication follow-ups; see [MODEL_LICENSES](../MODEL_LICENSES.md). Echelon and
dataset terms remain separate.
