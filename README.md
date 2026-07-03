# Lumen Quantum

Lumen Quantum ist ein kleines Python-Projekt zum Trainieren eines eigenen deutschsprachigen Decoder-only-Sprachmodells. Diese erste Version ist absichtlich nur ein Smoke-Test: Sie soll beweisen, dass Tokenizer, Datenvorbereitung, Training, Checkpoints, Resume, Generierung und Evaluation funktionieren.

Wichtig: Das Modell lädt niemals vortrainierte Modellgewichte. `LlamaForCausalLM` wird direkt aus `LlamaConfig` erstellt und startet mit zufälligen Gewichten.

## Projektstruktur

```text
configs/          YAML-Konfigurationen
data/raw/         lokale Trainingsdaten als .txt
data/processed/   Split- und Daten-Metadaten
data/tokenized/   tokenisierte Trainingsdaten
data/evals/       einfache Eval-Prompts und Ergebnisse
tokenizer/        selbst trainierte Tokenizer
scripts/          Pipeline-Skripte
models/           lokale Checkpoints und finale Smoke-Modelle
tests/            Pytest-Tests
docs/             spaetere Dokumentation
```

## 1. Einrichtung unter Windows PowerShell

Empfohlen ist Python 3.12 oder 3.11. Nutze fuer PyTorch nicht blind eine sehr neue Python-Version, falls dafuer noch keine passenden Wheels verfuegbar sind.

```powershell
cd C:\LumenQuantum
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Falls du nur Python 3.11 installiert hast:

```powershell
py -3.11 -m venv .venv
```

Falls PowerShell die Aktivierung blockiert:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## 2. Einrichtung unter Linux oder Cloud-GPU

```bash
cd /path/to/LumenQuantum
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Falls deine Cloud-Umgebung Python 3.11 nutzt:

```bash
python3.11 -m venv .venv
```

Prüfe optional, ob CUDA sichtbar ist:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 3. Beispiel-Trainingsdaten erstellen

Windows PowerShell:

```powershell
@"
Lumen ist ein deutschsprachiger Assistent.
Er antwortet ruhig, klar und hilfreich.

Die Sonne scheint ueber der Stadt.
Ein kleines Modell lernt zuerst nur den Ablauf.

Frage: Was ist Lumen?
Antwort: Lumen ist ein lokaler Testassistent.
"@ | Set-Content -Encoding UTF8 data\raw\beispiel.txt
```

Linux Bash:

```bash
cat > data/raw/beispiel.txt <<'EOF'
Lumen ist ein deutschsprachiger Assistent.
Er antwortet ruhig, klar und hilfreich.

Die Sonne scheint ueber der Stadt.
Ein kleines Modell lernt zuerst nur den Ablauf.

Frage: Was ist Lumen?
Antwort: Lumen ist ein lokaler Testassistent.
EOF
```

## 4. Tokenizer trainieren

```powershell
python scripts\train_tokenizer.py --config configs\smoke_5m.yaml
```

Linux:

```bash
python scripts/train_tokenizer.py --config configs/smoke_5m.yaml
```

Der Tokenizer wird in `tokenizer/smoke/` gespeichert.

## 5. Daten vorbereiten

Windows:

```powershell
python scripts\prepare_data.py --config configs\smoke_5m.yaml
```

Linux:

```bash
python scripts/prepare_data.py --config configs/smoke_5m.yaml
```

Die Dateien `data/tokenized/train.pt`, `data/tokenized/validation.pt`, `data/tokenized/test.pt` (Held-out) und Metadaten werden erzeugt. Der Test-Split wird während des Trainings nie geladen. Bei sehr kleinen Datensätzen kann `test.pt` entfallen (Warnung im Log).

## 6. Smoke-Modell trainieren

Windows:

```powershell
python scripts\train_smoke.py --config configs\smoke_5m.yaml
```

Linux:

```bash
python scripts/train_smoke.py --config configs/smoke_5m.yaml
```

Für einen sehr kurzen Funktionstest:

```powershell
python scripts\train_smoke.py --config configs\smoke_5m.yaml --max-steps 5
```

Checkpoints landen unter `models/smoke/checkpoints/checkpoint-step-XXXXX/`, das finale Modell unter `models/smoke/final/`. Auch ein kurzer Lauf mit `--max-steps 5` speichert am Ende einen vollständigen Resume-Checkpoint.

## 7. Training fortsetzen

Windows:

```powershell
python scripts\train_smoke.py --config configs\smoke_5m.yaml --max-steps 5
python scripts\train_smoke.py --config configs\smoke_5m.yaml --resume-from auto
```

Linux:

```bash
python scripts/train_smoke.py --config configs/smoke_5m.yaml --max-steps 5
python scripts/train_smoke.py --config configs/smoke_5m.yaml --resume-from auto
```

Du kannst auch einen konkreten Checkpoint-Ordner angeben, zum Beispiel `models/smoke/checkpoints/checkpoint-step-00005`.

## 8. Text generieren

Windows:

```powershell
python scripts\generate.py --checkpoint models\smoke\final --prompt "Lumen ist"
```

Linux:

```bash
python scripts/generate.py --checkpoint models/smoke/final --prompt "Lumen ist"
```

Optionen:

```bash
python scripts/generate.py --prompt "Frage: Was ist Lumen? Antwort:" --max-new-tokens 60 --temperature 0.8 --top-p 0.9
```

## 9. Einfache Evaluation ausführen

Lege zuerst Prompts an.

Windows:

```powershell
@"
Lumen ist
Frage: Was ist Lumen? Antwort:
Erklaere kurz, was ein Smoke-Test ist:
"@ | Set-Content -Encoding UTF8 data\evals\prompts.txt
python scripts\evaluate.py --checkpoint models\smoke\final --eval-file data\evals\prompts.txt
```

Linux:

```bash
cat > data/evals/prompts.txt <<'EOF'
Lumen ist
Frage: Was ist Lumen? Antwort:
Erklaere kurz, was ein Smoke-Test ist:
EOF
python scripts/evaluate.py --checkpoint models/smoke/final --eval-file data/evals/prompts.txt
```

Die Ergebnisse werden als JSONL in `data/evals/smoke_results.jsonl` gespeichert.

## 10. Tests ausführen

Windows:

```powershell
python -m pytest
```

Linux:

```bash
python -m pytest
```

## Hinweise

- Nutze fuer echtes Training mehr und bessere deutsche Textdaten als das Mini-Beispiel.
- Die Smoke-Konfiguration ist klein und dient nur dem Pipeline-Test.
- Wenn du die Tokenizer-Vokabulargroesse aenderst, bereite die Daten neu vor und trainiere das Modell neu.
- Lokale Checkpoints sind erlaubt. Verboten ist das Laden vortrainierter Modellgewichte aus externen Quellen.
