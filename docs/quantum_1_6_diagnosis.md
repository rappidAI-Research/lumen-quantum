# rappidAI Quantum — quantum-1.6-pilot generation diagnosis

> Historical diagnostic procedure from the earlier Lumen project phase. The
> current project identity is rappidAI Quantum; results remain experiment-specific.

Diese Diagnose klaert reproduzierbar, ob schlechte Antworten aus dem
PyTorch-Checkpoint, dem GGUF/llama.cpp-Pfad oder der Android-App-Inferenz
kommen. Sie veraendert keine Gewichte, keinen Tokenizer, keine Trainingsdaten,
keine GGUF-Datei und keine Android-App-Artefakte.

Wichtiger Ausgangsbefund: Terminaltests mit einfachen Prompts wie `Lumen ist`
und `Berlin ist` erzeugten normale deutsche Saetze. Die Saetze waren teilweise
faktisch unsinnig, aber keine zufaelligen Zeichenfolgen oder Zahlenmuster. Die
hoechste Prioritaet ist daher der Vergleich zwischen funktionierender
llama.cpp-Terminalausgabe und Android-Ausgabe mit identischer GGUF-Datei.

## Artefakte

```text
PyTorch:  models/quantum-1.6-pilot/final
Tokenizer: tokenizer/quantum-1
GGUF:     exports/quantum-1.6-pilot-v1.6.0-f16.gguf
llama.cpp tools/llama.cpp
Reports:  data/diagnostics/quantum-1.6-pilot/
```

Feste Prompts:

```text
Berlin ist
Ein Computer ist
Die deutsche Sprache
Nutzer: Hallo
Lumen:
```

Generierungsmodi:

```text
greedy_deterministic: temperature=0.0, top_p=1.0, top_k=0, max_new_tokens=64
controlled_sampling: temperature=0.7, top_p=0.9, top_k=40, max_new_tokens=64
```

## Windows-Tests

```powershell
cd C:\path\to\rappidai-quantum
.\.venv\Scripts\Activate.ps1
python -m pytest tests\test_diagnose_quantum_generation.py
python -m pytest
```

## RunPod PyTorch-Diagnose

```bash
cd /workspace/LumenQuantum
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/diagnose_quantum_generation.py \
  --config configs/quantum_1_6_diagnosis.yaml \
  --mode pytorch \
  --device cuda
```

Ausgaben:

```text
data/diagnostics/quantum-1.6-pilot/pytorch_generation_report.json
data/diagnostics/quantum-1.6-pilot/tokenizer_roundtrip_report.json
```

Der PyTorch-Modus prueft vor der Generierung:

- `tokenizer.model` im Modellordner ist byte-identisch zu `tokenizer/quantum-1/tokenizer.model`
- Vocab size stimmt
- Parameterzahl ist `49,295,872`
- Embedding und `lm_head` passen zum Tokenizer
- Tokenizer-Roundtrip ist stabil

## RunPod GGUF-Diagnose

Vorher sicherstellen, dass llama.cpp gebaut ist und ein Binary wie
`tools/llama.cpp/build/bin/llama-cli` existiert.

```bash
cd /workspace/LumenQuantum
source .venv/bin/activate

python scripts/diagnose_quantum_generation.py \
  --config configs/quantum_1_6_diagnosis.yaml \
  --mode gguf
```

Ausgabe:

```text
data/diagnostics/quantum-1.6-pilot/gguf_generation_report.json
```

Danach PyTorch und GGUF vergleichen:

```bash
python scripts/diagnose_quantum_generation.py \
  --config configs/quantum_1_6_diagnosis.yaml \
  --mode compare
```

Ausgabe:

```text
data/diagnostics/quantum-1.6-pilot/diagnosis_summary.md
data/diagnostics/quantum-1.6-pilot/diagnosis_comparison.json
```

## Android-Pruefung

Ziel: exakt dieselbe GGUF-Datei, denselben Prompt, dieselben Samplingwerte und
soweit moeglich denselben Seed verwenden wie im llama.cpp-Terminaltest.

Android muss pro Prompt und Modus einen JSONL-Record nach
`data/diagnostics/quantum-1.6-pilot/android_capture.jsonl` liefern. Ein Template
wird so erzeugt:

```bash
python scripts/diagnose_quantum_generation.py \
  --config configs/quantum_1_6_diagnosis.yaml \
  --mode android
```

Das Template liegt danach hier:

```text
data/diagnostics/quantum-1.6-pilot/android_capture_template.json
```

Pflichtfelder je Android-Record:

```json
{
  "model_id": "quantum-1.6-pilot-v1.6.0-f16",
  "ui_active_model_id": "quantum-1.6-pilot-v1.6.0-f16",
  "local_model_path": "/data/user/0/<package>/files/models/quantum-1.6-pilot-v1.6.0-f16.gguf",
  "file_size_bytes": 123,
  "sha256": "sha256-der-tatsaechlich-geladenen-gguf-datei",
  "prompt": "Berlin ist",
  "mode": "controlled_sampling",
  "sampling": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "seed": 20260705,
    "max_tokens": 64,
    "context_length": 512
  },
  "raw_stream_fragments": ["Berlin", " ist", " ..."],
  "raw_stream_text": "Berlin ist ...",
  "final_ui_text": "Berlin ist ..."
}
```

Die App muss vor UI-Nachbearbeitung die rohen gestreamten Token-Fragmente
loggen. Wichtig ist auch, dass die App die SHA-256 und Dateigroesse der
tatsaechlich geladenen GGUF-Datei loggt, nicht nur die erwartete Modell-ID.

Nach Import des Android-Logs:

```bash
python scripts/diagnose_quantum_generation.py \
  --config configs/quantum_1_6_diagnosis.yaml \
  --mode android

python scripts/diagnose_quantum_generation.py \
  --config configs/quantum_1_6_diagnosis.yaml \
  --mode compare
```

Der Android-Modus markiert:

- falsche oder alte GGUF-Datei durch SHA-256-Mismatch
- unvollstaendige Datei durch Dateigroessen-Mismatch
- UI-Modell stimmt nicht mit geladenem Modell ueberein
- Prompt wurde vor der Uebergabe veraendert
- Samplingwerte fehlen oder weichen ab
- rohe Stream-Fragmente fehlen

## Komplettlauf auf RunPod

```bash
cd /workspace/LumenQuantum
source .venv/bin/activate

python scripts/diagnose_quantum_generation.py \
  --config configs/quantum_1_6_diagnosis.yaml \
  --mode all \
  --device cuda
```

Wenn noch kein Android-Capture vorhanden ist, erzeugt `--mode all` trotzdem den
Android-Template-Report und vermerkt `awaiting_android_capture`.
