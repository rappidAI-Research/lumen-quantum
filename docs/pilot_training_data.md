# rappidAI Quantum — quantum-1 pilot training data

> Historical pilot procedure from the earlier Lumen project phase. The current
> project identity is rappidAI Quantum; this document is retained for provenance.

Diese Datei beschreibt den Tokenisierungsschritt fuer den spaeteren
Cloud-Pilot von `quantum-1-base`. Es wird noch kein Modell trainiert und kein
Cloud-Server gestartet.

## Eingaben

Die Tokenisierung erwartet die bereinigten, bereits getrennten Splits:

- `data/quantum/cleaned/train.jsonl`
- `data/quantum/cleaned/validation.jsonl`
- `data/quantum/cleaned/test.jsonl`

Der lokale Tokenizer muss vorhanden sein:

- `tokenizer/quantum-1-pilot/tokenizer.model`
- `tokenizer/quantum-1-pilot/tokenizer_manifest.json`

Es werden keine vortrainierten Tokenizer und keine Modellgewichte geladen.

## Ausgabe

Die Ausgabe liegt in:

- `data/quantum/tokenized/pilot/train.pt`
- `data/quantum/tokenized/pilot/validation.pt`
- `data/quantum/tokenized/pilot/test.pt`
- `data/quantum/tokenized/pilot/tokenization_manifest.json`

Diese Dateien gehoeren nicht in Git. Sie sind reproduzierbare Datenartefakte
und koennen aus den Cleaned-Daten plus Tokenizer neu erstellt werden.

## Packing-Regeln

- Jedes Dokument wird separat tokenisiert.
- Nach jedem Dokument wird `</s>` angehaengt.
- Train, Validation und Test werden getrennt gelesen und niemals vermischt.
- Tokens werden zu Sequenzen mit exakt 512 Tokens gepackt.
- Die letzte unvollstaendige Sequenz wird mit `<pad>` aufgefuellt.
- Padding-Positionen erhalten in `labels` den Wert `-100`.
- Resttokens und Padding werden im Manifest dokumentiert.

## Windows PowerShell

```powershell
cd C:\path\to\rappidai-quantum
.\.venv\Scripts\Activate.ps1
python scripts\tokenize_quantum_data.py --config configs\quantum_1_pilot_data.yaml
```

Wenn du bewusst neu erzeugen willst:

```powershell
python scripts\tokenize_quantum_data.py --config configs\quantum_1_pilot_data.yaml --overwrite
```

Kleiner Limit-Test:

```powershell
python scripts\tokenize_quantum_data.py --config configs\quantum_1_pilot_data.yaml --overwrite --max-tokens 100000 --validation-max-tokens 20000 --test-max-tokens 20000
```

## Linux oder RunPod

```bash
cd /workspace/LumenQuantum
source .venv/bin/activate
python scripts/tokenize_quantum_data.py --config configs/quantum_1_pilot_data.yaml
```

Wenn du bewusst neu erzeugen willst:

```bash
python scripts/tokenize_quantum_data.py --config configs/quantum_1_pilot_data.yaml --overwrite
```

Kleiner Limit-Test:

```bash
python scripts/tokenize_quantum_data.py --config configs/quantum_1_pilot_data.yaml --overwrite --max-tokens 100000 --validation-max-tokens 20000 --test-max-tokens 20000
```

## Manifest

`tokenization_manifest.json` dokumentiert:

- Tokenizer-Hash
- Hashes der Eingabedateien
- Kontextlaenge
- Dokumentanzahlen je Split
- Tokenzahlen je Split
- Sequenzzahlen je Split
- verwendete Limits
- Seed
- Erstellungsdatum
- Resttoken-/Padding-Policy

## Gitignore

Diese Ordner duerfen nicht versioniert werden:

- `data/quantum/raw/`
- `data/quantum/cleaned/`
- `data/quantum/tokenized/`
- `data/quantum/manifests/`
- `data/quantum/reports/`
- `models/`
- `tokenizer/`
