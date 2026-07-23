# rappidAI Quantum — historical quantum-1 final pipeline plan

> Historical plan from the earlier Lumen project phase. It is not a published
> production release plan and is retained for provenance.

Der Cloud-Pilot bleibt archiviert und wird nicht veraendert. Die finale v1
nutzt neue Ausgabeorte:

```text
data/quantum/final/
tokenizer/quantum-1/
models/quantum-1-base/
```

Es werden keine vortrainierten Modellgewichte und keine vortrainierten
Tokenizer verwendet. `quantum-1-base` wird weiterhin direkt aus
`LlamaConfig` mit zufaelligen Gewichten initialisiert.

## Ziele

- Sprache: Deutsch
- Kontextlaenge: 512
- Train: ungefaehr 100,000,000 Tokens
- Validation: ungefaehr 1,000,000 Tokens
- Test: ungefaehr 1,000,000 Tokens
- Tokenizer: SentencePiece-BPE, 16,384 Tokens
- Modell: `quantum-1-base`, Zielbereich 45 bis 55 Mio. Parameter

Die exakten Tokenzahlen werden erst nach Training des finalen Tokenizers und
nach `scripts/tokenize_quantum_data.py` in
`data/quantum/final/tokenized/tokenization_manifest.json` und
`data/quantum/final/reports/final_data_validation_report.json` berichtet.

## Daten

Die bestehende Quelle bleibt:

```text
epfml/FineWeb2-HQ / deu_Latn / train
```

Der Download nutzt Hugging Face Datasets im Streaming-Modus und stoppt durch
die Limits in `configs/quantum_1_final_data.yaml`. Es wird nicht der komplette
Datensatz heruntergeladen.

Ausgabe:

```text
data/quantum/final/raw/
data/quantum/final/cleaned/
data/quantum/final/manifests/
data/quantum/final/reports/
```

Cleaning entfernt leere Texte, sehr kurze Texte, exakte Duplikate, kaputte
Texte, Texte mit extremen Zeichenwiederholungen, zu viele Nicht-Buchstaben und
offensichtliches Boilerplate. Train, Validation und Test werden per stabiler
SHA256-Bucket-Logik getrennt.

## Tokenizer

Der finale Tokenizer wird nur aus
`data/quantum/final/cleaned/train.jsonl` trainiert. Validation und Test duerfen
nicht in das Tokenizer-Training einfließen.

Special Tokens in stabiler Reihenfolge:

```text
<unk>          0
<s>            1
</s>           2
<pad>          3
<|system|>     4
<|user|>       5
<|assistant|>  6
```

Nach erfolgreicher Validierung schreibt
`scripts/validate_final_tokenizer.py` die Dateien `FINAL_FROZEN` und
`freeze_manifest.json`. `scripts/train_quantum_tokenizer.py` verweigert danach
ein erneutes Schreiben in diesen Ordner.

## Tokenisierung

`scripts/tokenize_quantum_data.py` nutzt
`configs/quantum_1_final_tokenizer.yaml` auch als Tokenisierungsconfig. Dabei
werden Train, Validation und Test getrennt gelesen, jedes Dokument mit `</s>`
abgeschlossen und feste Sequenzen mit exakt 512 Tokens erzeugt.

Ausgabe:

```text
data/quantum/final/tokenized/train.pt
data/quantum/final/tokenized/validation.pt
data/quantum/final/tokenized/test.pt
data/quantum/final/tokenized/tokenization_manifest.json
```

## Training

`configs/quantum_1_final_train.yaml` bereitet Cloud-Training vor, startet aber
standardmaessig nichts:

```yaml
training:
  output_dir: "models/quantum-1-base"
  max_steps: 0
  mixed_precision: "bf16"
  resume_from_checkpoint: null
```

Checkpoints, Resume und Evaluation sind in der Config vorgesehen. Pilotgewichte
duerfen nicht uebernommen werden, weil der finale Tokenizer neu ist.

## Windows-Befehle

```powershell
cd C:\path\to\rappidai-quantum
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts\download_quantum_data.py --config configs\quantum_1_final_data.yaml
python scripts\clean_quantum_data.py --config configs\quantum_1_final_data.yaml
python scripts\sample_quantum_data.py --config configs\quantum_1_final_data.yaml
python scripts\inspect_quantum_data.py --config configs\quantum_1_final_data.yaml
python scripts\build_data_manifest.py --config configs\quantum_1_final_data.yaml
python scripts\validate_final_data.py --data-config configs\quantum_1_final_data.yaml --skip-tokenized

python scripts\train_quantum_tokenizer.py --config configs\quantum_1_final_tokenizer.yaml
python scripts\validate_final_tokenizer.py --config configs\quantum_1_final_tokenizer.yaml

python scripts\tokenize_quantum_data.py --config configs\quantum_1_final_tokenizer.yaml
python scripts\validate_final_data.py --data-config configs\quantum_1_final_data.yaml --tokenization-config configs\quantum_1_final_tokenizer.yaml --require-tokenized

python scripts\inspect_model_size.py --config configs\quantum_1_final_train.yaml
python scripts\validate_quantum_model.py --config configs\quantum_1_final_train.yaml
```

## RunPod-Befehle

```bash
cd /workspace/LumenQuantum
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/download_quantum_data.py --config configs/quantum_1_final_data.yaml
python scripts/clean_quantum_data.py --config configs/quantum_1_final_data.yaml
python scripts/sample_quantum_data.py --config configs/quantum_1_final_data.yaml
python scripts/inspect_quantum_data.py --config configs/quantum_1_final_data.yaml
python scripts/build_data_manifest.py --config configs/quantum_1_final_data.yaml
python scripts/validate_final_data.py --data-config configs/quantum_1_final_data.yaml --skip-tokenized

python scripts/train_quantum_tokenizer.py --config configs/quantum_1_final_tokenizer.yaml
python scripts/validate_final_tokenizer.py --config configs/quantum_1_final_tokenizer.yaml

python scripts/tokenize_quantum_data.py --config configs/quantum_1_final_tokenizer.yaml
python scripts/validate_final_data.py --data-config configs/quantum_1_final_data.yaml --tokenization-config configs/quantum_1_final_tokenizer.yaml --require-tokenized

python scripts/inspect_model_size.py --config configs/quantum_1_final_train.yaml
python scripts/validate_quantum_model.py --config configs/quantum_1_final_train.yaml
```

## Nicht in Git

Diese Artefakte bleiben lokal oder auf RunPod:

```text
data/quantum/final/
tokenizer/quantum-1/
models/quantum-1-base/
logs/
```

Versioniert werden nur Configs, Skripte, Tests und Doku.
