# rappidAI Quantum — quantum-1 Cloud-GPU pilot

> Historical pilot procedure from the earlier Lumen project phase. The current
> project identity is rappidAI Quantum; this document is retained for provenance.

Diese Anleitung bereitet den ersten kurzen Cloud-Pilotlauf fuer
`quantum-1-base` vor. Es wird keine Architektur geaendert, kein Tokenizer neu
trainiert und keine Datenpipeline neu gebaut.

## Ziel

- Modell: `quantum-1-base`
- Parameter: `49,295,872`
- Tokenizer: `tokenizer/quantum-1-pilot/`
- Eingabe: `data/quantum/tokenized/pilot/`
- Kontextlaenge: 512 Tokens
- GPU-Ziel: RTX 3090 oder RTX 4090 mit 24 GB VRAM
- Pilotlimit: maximal 100 Trainingsschritte

## Vor dem Cloud-Lauf

Diese Dateien muessen auf der Maschine vorhanden sein:

- `tokenizer/quantum-1-pilot/tokenizer.model`
- `tokenizer/quantum-1-pilot/tokenizer_manifest.json`
- `data/quantum/tokenized/pilot/train.pt`
- `data/quantum/tokenized/pilot/validation.pt`
- `data/quantum/tokenized/pilot/test.pt`
- `data/quantum/tokenized/pilot/tokenization_manifest.json`

## Lokaler CPU-Dry-Run

Windows PowerShell:

```powershell
cd C:\path\to\rappidai-quantum
.\.venv\Scripts\Activate.ps1
python scripts\train_quantum_pilot.py --config configs\quantum_1_cloud_pilot.yaml --dry-run
```

Der Dry-Run baut das Modell, laedt die tokenisierten Daten und macht einen
Forward Pass. Es werden keine Gewichte dauerhaft trainiert oder gespeichert.

## RunPod/Linux vorbereiten

```bash
cd /workspace/LumenQuantum
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Dry-Run auf RunPod/Linux:

```bash
cd /workspace/LumenQuantum
source .venv/bin/activate
python scripts/train_quantum_pilot.py --config configs/quantum_1_cloud_pilot.yaml --dry-run
```

## GPU-Pilot mit 100 Schritten

```bash
cd /workspace/LumenQuantum
source .venv/bin/activate
python scripts/train_quantum_pilot.py --config configs/quantum_1_cloud_pilot.yaml
```

Das Skript gibt beim Start aus:

- verwendetes Geraet
- GPU-Name
- VRAM
- PyTorch-Version
- CUDA-Version
- verwendete Mixed Precision

Mixed Precision steht in der YAML auf `auto`: CUDA nutzt bevorzugt `bf16`,
falls unterstuetzt, sonst `fp16`. CPU nutzt `no`.

## Resume

```bash
cd /workspace/LumenQuantum
source .venv/bin/activate
python scripts/train_quantum_pilot.py --config configs/quantum_1_cloud_pilot.yaml --resume-from auto
```

Checkpoints liegen unter:

```text
models/quantum-1-cloud-pilot/checkpoints/checkpoint-step-XXXXX/
```

Jeder Checkpoint enthaelt Modellgewichte, Optimizer, Scheduler, global_step,
Epoch, RNG-Zustand, Config und Tokenizer-Manifest.

## Nicht in Git

Diese Ordner bleiben lokale oder Cloud-Artefakte und gehoeren nicht ins Repo:

- `data/quantum/tokenized/`
- `models/`
- `tokenizer/`
- `logs/`
