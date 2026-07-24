# Licensing boundaries

## Source code and documentation

Original repository source code, configurations, tests, and project
documentation are licensed under Apache License 2.0, which is now present on
`main`. Git history attributes all tracked original material to the maintainer
account, and the audit found no vendored copied source. The maintainer must
still confirm that they own or can license every original tracked contribution;
merging the license files did not, by itself, make that legal determination. See
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

## Approval checklist

The Apache-2.0 files are merged, but the ownership determination remains a human
decision. The maintainer must confirm:

1. they own or have permission to license all original tracked code and prose;
2. no employer, client, collaborator, or contract holds conflicting rights;
3. no copied snippet lacks its source, notice, or compatible license;
4. generated reports contain no redistributable third-party content beyond what
   the relevant terms permit; and
5. Apache-2.0 is the intended license for future inbound contributions.

Model, tokenizer, and dataset license decisions remain separate even after this
checklist is approved.
