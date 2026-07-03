# Lumen Quantum

Lumen Quantum ist ein kleines Python-Projekt zum Trainieren eines eigenen deutschsprachigen Decoder-only-Sprachmodells. Diese Version ist ein Smoke-Test: Sie prueft Tokenizer, Datenvorbereitung, Training, Checkpoints, Resume, Generierung, Evaluation und GGUF-Export.

Wichtig: Das Modell laedt niemals vortrainierte Modellgewichte. `LlamaForCausalLM` wird direkt aus `LlamaConfig` erstellt und startet mit zufaelligen Gewichten. Der Tokenizer wird ebenfalls selbst aus lokalen Textdaten trainiert.

## Projektstruktur

```text
configs/          YAML-Konfigurationen
data/raw/         lokale Trainingsdaten als .txt
data/processed/   Split- und Daten-Metadaten
data/tokenized/   tokenisierte Trainingsdaten
data/evals/       einfache Eval-Prompts und Ergebnisse
tokenizer/        selbst trainierte Tokenizer
scripts/          Pipeline-Skripte
models/           lokale Checkpoints, finale Modelle und GGUF-Dateien
tests/            Pytest-Tests
docs/             spaetere Dokumentation
```

## 1. Einrichtung unter Windows PowerShell

Empfohlen ist Python 3.12 oder 3.11.

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

## 2. Einrichtung unter Linux oder Cloud-GPU

```bash
cd /path/to/LumenQuantum
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Falls deine Umgebung Python 3.11 nutzt:

```bash
python3.11 -m venv .venv
```

Optional CUDA pruefen:

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

## 4. Alte Smoke-Artefakte entfernen

Wenn du bereits einen Smoke-Lauf mit dem alten Byte-Level-BPE-Tokenizer gemacht hast, loesche die abgeleiteten Artefakte. Die Rohdaten in `data/raw/` bleiben erhalten.

Windows PowerShell:

```powershell
Remove-Item -Recurse -Force tokenizer\smoke, data\tokenized\*, data\processed\*, models\smoke -ErrorAction SilentlyContinue
```

Linux:

```bash
rm -rf tokenizer/smoke data/tokenized/* data/processed/* models/smoke
```

Die neue Reihenfolge ist:

```text
Tokenizer neu trainieren -> Daten neu tokenisieren -> Smoke-Modell neu trainieren -> GGUF exportieren
```

## 5. Tokenizer trainieren

Der Smoke-Tokenizer ist ein selbst trainierter SentencePiece-BPE-Tokenizer. Das ist wichtig fuer llama.cpp/GGUF, weil der klassische LLaMA-Konverter `tokenizer.model` erwartet. Die Special Tokens sind LLaMA-kompatibel: `<unk>` = 0, `<s>` = 1, `</s>` = 2, `<pad>` = 3. Byte-Fallback bleibt fuer diesen Smoke-Test aus.

Windows:

```powershell
python scripts\train_tokenizer.py --config configs\smoke_5m.yaml
```

Linux:

```bash
python scripts/train_tokenizer.py --config configs/smoke_5m.yaml
```

Ergebnis: `tokenizer/smoke/tokenizer.model` plus zugehoerige Konfigurationsdateien.

## 6. Daten vorbereiten

Windows:

```powershell
python scripts\prepare_data.py --config configs\smoke_5m.yaml
```

Linux:

```bash
python scripts/prepare_data.py --config configs/smoke_5m.yaml
```

Die Dateien `data/tokenized/train.pt`, `data/tokenized/validation.pt`, optional `data/tokenized/test.pt` und Metadaten werden erzeugt.

## 7. Smoke-Modell trainieren

Windows:

```powershell
python scripts\train_smoke.py --config configs\smoke_5m.yaml
```

Linux:

```bash
python scripts/train_smoke.py --config configs/smoke_5m.yaml
```

Sehr kurzer Funktionstest:

```powershell
python scripts\train_smoke.py --config configs\smoke_5m.yaml --max-steps 5
```

Checkpoints landen unter `models/smoke/checkpoints/checkpoint-step-XXXXX/`, das finale Modell unter `models/smoke/final/`. Das finale Modell enthaelt `config.json`, Modellgewichte und `tokenizer.model`.

## 8. Training fortsetzen

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

## 9. Text generieren

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

## 10. GGUF exportieren

Der Export nutzt das lokal erzeugte Hugging-Face-Modell aus `models/smoke/final/` und laedt keine vortrainierten Gewichte. Der Wrapper prueft vorher `config.json`, Modellgewichte und `tokenizer.model` und ruft dann den llama.cpp-Konverter auf.

Windows PowerShell, wenn llama.cpp lokal unter `C:\srv\lumen\llama.cpp` liegt:

```powershell
python scripts\export_gguf.py --model-dir models\smoke\final --output-file models\smoke\quantum-smoke-f16.gguf --llama-cpp-dir C:\srv\lumen\llama.cpp
```

Linux oder Raspberry Pi mit llama.cpp unter `/srv/lumen/llama.cpp`:

```bash
python scripts/export_gguf.py --model-dir models/smoke/final --output-file models/smoke/quantum-smoke-f16.gguf --llama-cpp-dir /srv/lumen/llama.cpp
```

Direkter llama.cpp-Konvertierungsbefehl:

```bash
python /srv/lumen/llama.cpp/convert_hf_to_gguf.py models/smoke/final --outfile models/smoke/quantum-smoke-f16.gguf --outtype f16
```

Nutze den Wrapper, wenn du vorher pruefen willst, ob `config.json`, Gewichte und `tokenizer.model` vollstaendig vorhanden sind.

Auf dem Raspberry Pi testest du das zufaellig initialisierte Smoke-Modell als reine Completion, nicht als Chat:

```bash
cd /srv/lumen/llama.cpp
./build/bin/llama-completion -m /srv/lumen/models/quantum-smoke-f16.gguf -p "Lumen ist" -n 32
```

`llama-cli` ist in aktuellen llama.cpp-Versionen ein Chat-Client. Ein zufaelliges Smoke-Modell wurde nicht auf Chat-Formate trainiert und kann deshalb mit `llama-cli` beim Parsen der Antwort abbrechen.

## 11. Einfache Evaluation ausfuehren

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

## 12. Tests ausfuehren

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
- Wenn du Tokenizer oder Vokabulargroesse aenderst, musst du Daten neu tokenisieren und das Modell neu trainieren.
- Lokale Checkpoints sind erlaubt. Verboten ist das Laden vortrainierter Modellgewichte aus externen Quellen.
