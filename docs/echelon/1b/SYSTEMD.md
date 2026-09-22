# Quantum 1 Echelon 1B — systemd deployment interface

Status: pre-production interface. This does not claim an AWS host has been
provisioned or that a disconnect/Spot recovery test has passed.

## Why systemd

The MacBook is not part of the training lifecycle. Losing local Internet,
closing the laptop or dropping SSH must not stop the authoritative process.
`tmux` remains useful for observation but is not the process supervisor.

## Files

- `ops/systemd/quantum-1-echelon@.service` — fixed service template.
- `ops/systemd/quantum-1-echelon.env.example` — non-secret example.
- `scripts/echelon_systemd_entrypoint.py` — strict environment parser.
- `scripts/echelon_unattended.py` — persistent log/status and wall-time guard.

The environment file accepts only five allowlisted keys and stores the child
command as a JSON argv array. It is never evaluated by a shell.

## Host layout

The first AWS smoke should use:

- repository: `/opt/rappidai/lumen-quantum`;
- virtual environment: `/opt/rappidai/lumen-quantum/.venv`;
- persistent status/logs: `/var/lib/rappidai/quantum-1-echelon`;
- active token shards/checkpoints: `/mnt/echelon`;
- recovery source of truth: private S3 through the EC2 instance role.

Do not place long-lived AWS access keys in `/etc/rappidai`, the repository,
logs, checkpoints or shell history.

## Installation on the future AWS host

After the host image, Linux user and repository checkout are explicitly
approved:

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin echelon
sudo install -d -o echelon -g echelon -m 0700 /var/lib/rappidai/quantum-1-echelon
sudo install -d -o echelon -g echelon -m 0700 /mnt/echelon
sudo install -d -m 0750 /etc/rappidai
sudo install -m 0644 ops/systemd/quantum-1-echelon@.service /etc/systemd/system/
sudo cp ops/systemd/quantum-1-echelon.env.example /etc/rappidai/quantum-1-echelon-smoke.env
sudo chmod 0600 /etc/rappidai/quantum-1-echelon-smoke.env
sudo systemctl daemon-reload
```

Edit the smoke environment file before starting. The tracked example references
paths that must exist on the host.

## Commands

```bash
sudo systemctl start quantum-1-echelon@smoke
sudo systemctl status quantum-1-echelon@smoke
sudo journalctl -u quantum-1-echelon@smoke -f
sudo systemctl stop quantum-1-echelon@smoke
```

The service intentionally has `Restart=no`. An automatic restart loop could
burn promotional credits or repeatedly load a corrupt checkpoint. Recovery is
an explicit verified action.

## Mandatory live acceptance test

Before the substantial H100 run:

1. start the bounded smoke through systemd;
2. disconnect SSH and take the MacBook offline;
3. reconnect later and prove the same server-side run continued;
4. create a locally integrity-verified checkpoint;
5. publish it to private S3 and confirm the recovery-valid marker;
6. stop/terminate the workload;
7. restore on replacement capacity from the verified checkpoint;
8. verify model, optimizer, scheduler, all RNG state, processed-token count and
   exact global token offset.

Passing unit tests is not a substitute for this live AWS failure test.
