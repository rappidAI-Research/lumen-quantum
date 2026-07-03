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

## 13. quantum-1 FineWeb2-HQ Datenpipeline vorbereiten

Diese Pipeline streamt eine kleine deutsche Pilotmenge aus `epfml/FineWeb2-HQ`, Subset `deu_Latn`. Es wird kein quantum-1-Tokenizer trainiert, kein grosses Modell trainiert und es werden keine Modellgewichte geladen.

Limits in `configs/quantum_1_data.yaml`:

```text
max_documents: 100000
max_raw_bytes: 2147483648
```

Die Ausgaben liegen hier:

```text
data/quantum/raw/
data/quantum/cleaned/
data/quantum/manifests/
data/quantum/reports/
```

Windows PowerShell:

```powershell
cd C:\LumenQuantum
pip install -r requirements.txt
python scripts\download_quantum_data.py --config configs\quantum_1_data.yaml
python scripts\clean_quantum_data.py --config configs\quantum_1_data.yaml
python scripts\sample_quantum_data.py --config configs\quantum_1_data.yaml
python scripts\inspect_quantum_data.py --config configs\quantum_1_data.yaml
python scripts\build_data_manifest.py --config configs\quantum_1_data.yaml
```

Linux oder spaeter RunPod:

```bash
cd /path/to/LumenQuantum
pip install -r requirements.txt
python scripts/download_quantum_data.py --config configs/quantum_1_data.yaml
python scripts/clean_quantum_data.py --config configs/quantum_1_data.yaml
python scripts/sample_quantum_data.py --config configs/quantum_1_data.yaml
python scripts/inspect_quantum_data.py --config configs/quantum_1_data.yaml
python scripts/build_data_manifest.py --config configs/quantum_1_data.yaml
```

Wichtige Dateien nach dem Lauf:

```text
data/quantum/raw/fineweb2_hq_deu_latn_raw.jsonl
data/quantum/cleaned/documents_cleaned.jsonl
data/quantum/cleaned/train.jsonl
data/quantum/cleaned/validation.jsonl
data/quantum/cleaned/test.jsonl
data/quantum/reports/quantum_data_report.json
data/quantum/manifests/data_manifest.json
data/quantum/manifests/data_manifest.md
```

Die Splits werden per stabiler SHA256-Bucket-Logik erzeugt: Train 98 %, Validation 1 %, Test 1 %. Exakte Duplikate und offensichtlich unbrauchbare Texte werden entfernt. Die Tokenzahlen sind nur grobe Schaetzungen, weil fuer quantum-1 noch kein Tokenizer trainiert wird.

## 14. quantum-1 Pilot-Tokenizer trainieren

Dieser Schritt baut nur den Pilot-Tokenizer `quantum-1-pilot`. Er ist noch nicht der finale eingefrorene quantum-1-Tokenizer und wird spaeter auf einer groesseren dokumentierten Datenmenge neu trainiert. Es werden keine vortrainierten Tokenizer und keine Modellgewichte geladen.

Der Pilot-Tokenizer wird ausschliesslich aus `data/quantum/cleaned/train.jsonl` trainiert. `validation.jsonl` und `test.jsonl` bleiben fuer Qualitaetspruefung reserviert.

Windows PowerShell:

```powershell
cd C:\LumenQuantum
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\train_quantum_tokenizer.py --config configs\quantum_1_tokenizer_pilot.yaml
python scripts\validate_quantum_tokenizer.py --config configs\quantum_1_tokenizer_pilot.yaml
```

Ergebnis:

```text
tokenizer/quantum-1-pilot/tokenizer.model
tokenizer/quantum-1-pilot/tokenizer.vocab
tokenizer/quantum-1-pilot/tokenizer_config.json
tokenizer/quantum-1-pilot/special_tokens_map.json
tokenizer/quantum-1-pilot/tokenizer_manifest.json
tokenizer/quantum-1-pilot/validation_report.json
```

Der Tokenizer nutzt SentencePiece BPE mit 16384 Tokens. Die LLaMA/GGUF-kompatiblen Basis-IDs bleiben fest: `<unk>` = 0, `<s>` = 1, `</s>` = 2, `<pad>` = 3. Fuer spaeteres Chat-Training sind ausserdem `<|system|>`, `<|user|>` und `<|assistant|>` enthalten.

Nur die Tokenizer-Tests ausfuehren:

```powershell
python -m pytest tests\test_quantum_tokenizer.py
```

Alle Tests ausfuehren:

```powershell
python -m pytest
```

## 15. quantum-1-base Architektur pruefen

Dieser Schritt baut noch kein grosses Modelltraining. Er prueft nur die echte `quantum-1-base` Pilot-Architektur mit zufaellig initialisierten Gewichten, den lokalen `quantum-1-pilot`-Tokenizer, einen CPU-Forward-Pass und temporaeres Speichern/Laden.

Die geplante Architektur steht in `configs/quantum_1_base_pilot.yaml`:

```text
hidden_size: 512
intermediate_size: 1536
num_hidden_layers: 12
num_attention_heads: 8
num_key_value_heads: 8
max_position_embeddings: 512
tie_word_embeddings: true
```

Mit dem Pilot-Tokenizer `vocab_size = 16384` hat das Modell exakt `49,295,872` Parameter. Es liegt damit im Zielbereich von 45 bis 55 Millionen Parametern.

Windows PowerShell:

```powershell
cd C:\LumenQuantum
.\.venv\Scripts\Activate.ps1
python scripts\inspect_model_size.py --config configs\quantum_1_base_pilot.yaml
python scripts\validate_quantum_model.py --config configs\quantum_1_base_pilot.yaml
python -m pytest tests\test_quantum_model.py
```

Optionaler Mini-Checkpoint-Test, kein grosses Training:

```powershell
python scripts\train_quantum_pilot.py --config configs\quantum_1_base_pilot.yaml --max-steps 1
```

Danach kann ein lokal gespeicherter Pilot-Checkpoint so getestet werden:

```powershell
python scripts\generate_quantum.py --config configs\quantum_1_base_pilot.yaml --checkpoint models\quantum-1-base\final --prompt "Lumen ist"
```

## Hinweise

- Nutze fuer echtes Training mehr und bessere deutsche Textdaten als das Mini-Beispiel.
- Die Smoke-Konfiguration ist klein und dient nur dem Pipeline-Test.
- Wenn du Tokenizer oder Vokabulargroesse aenderst, musst du Daten neu tokenisieren und das Modell neu trainieren.
- Lokale Checkpoints sind erlaubt. Verboten ist das Laden vortrainierter Modellgewichte aus externen Quellen.
