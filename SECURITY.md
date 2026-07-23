# Security policy

## Supported code

Only the latest `main` revision and an explicitly published release, if one
exists, are eligible for security fixes. Experimental models and datasets carry
no production-security warranty.

## Private reporting

Do not open a public issue for a vulnerability, credential, personal data, or
restricted artifact. Use GitHub private vulnerability reporting when it is
available, or email `cikemgil@rappidai-research.com` with:

- affected revision and component;
- impact and realistic threat model;
- minimal reproduction without secrets or restricted data; and
- any suggested mitigation.

No response-time or disclosure deadline is guaranteed. The maintainer will
acknowledge and coordinate remediation as capacity permits.

## ML-specific risks

Treat checkpoints, pickle-based training state, datasets, tokenizers, and
third-party executables as untrusted. `torch.load(..., weights_only=False)` is
used only for locally produced resume state and can execute malicious pickle
payloads. Never resume from an untrusted checkpoint. Do not run downloaded
llama.cpp binaries or conversion scripts without verifying their origin and
revision.
