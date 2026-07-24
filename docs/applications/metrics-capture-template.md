# Application metrics capture template

Capture these values **manually, immediately before submitting** the OpenAI
Codex for Open Source application. Do not reuse older numbers as current facts,
and do not round or inflate anything.

## Capture

| Metric | Value | Source URL | Notes |
|---|---:|---|---|
| Capture date and time (UTC) | _fill_ | — | Exact capture time |
| Repository stars | _fill_ | repository page | |
| Forks | _fill_ | repository page | |
| Open issues | _fill_ | issues page | |
| Closed issues | _fill_ | issues page (filter closed) | |
| Open pull requests | _fill_ | pull requests page | Includes Dependabot PRs |
| External contributors | _fill_ | contributors page | Exclude the sole maintainer and bots |
| Releases | _fill_ | releases page | |
| GitHub usage signals | _fill_ | traffic/insights | Only if meaningful and verifiable |
| Hugging Face downloads | _fill_ | HF model pages | Only if public and verifiable |
| Package downloads | _fill_ | PyPI (if published) | "None" if not published |
| Other verified adoption | _fill_ | — | Only verifiable signals |

## Rules

- If a value is zero, write `0`. Do not omit it to look better.
- If a value cannot be verified, write `Not verified` and cite why.
- Do not describe the project as widely adopted, team-run, or institutionally
  backed.
- Update both the values and the capture date in
  [openai-codex-for-oss.md](openai-codex-for-oss.md) before submission.
