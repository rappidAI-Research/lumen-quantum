# Lumen quantum-1-base Modell-Spezifikation

`quantum-1-base` ist die vorbereitete Basisarchitektur fuer das spaetere
deutschsprachige Lumen-Modell. In dieser Phase wird noch kein grosses Training
gestartet. Ziel ist nur, Architektur, Tokenizer-Kompatibilitaet, Forward Pass
und lokales Speichern zu pruefen.

## Grundregeln

- Keine vortrainierten Modellgewichte laden.
- Kein `from_pretrained` fuer Modelle verwenden.
- `LlamaForCausalLM` wird direkt aus `LlamaConfig` erstellt.
- Der Tokenizer ist der lokale Pilot-Tokenizer `tokenizer/quantum-1-pilot/`.
- Der Tokenizer muss spaeter durch den final eingefrorenen quantum-1-Tokenizer ersetzt werden.
- Das Modellverzeichnis soll HF-kompatibel und damit fuer llama.cpp/GGUF exportierbar sein.

## Architektur

| Feld | Wert |
| --- | ---: |
| `vocab_size` | automatisch aus `tokenizer/quantum-1-pilot/` |
| `hidden_size` | 512 |
| `intermediate_size` | 1536 |
| `num_hidden_layers` | 12 |
| `num_attention_heads` | 8 |
| `num_key_value_heads` | 8 |
| `max_position_embeddings` | 512 |
| `tie_word_embeddings` | true |
| `attention_dropout` | 0.0 |

Mit `vocab_size = 16384` ergibt diese Architektur durch geteilte Input-/Output-
Embeddings exakt `49,295,872` Parameter und liegt damit im Zielbereich von
45 bis 55 Millionen Parametern.

## Tokenizer-Anforderungen

Die klassischen LLaMA/GGUF-IDs muessen stimmen:

| Token | ID |
| --- | ---: |
| `<unk>` | 0 |
| `<s>` | 1 |
| `</s>` | 2 |
| `<pad>` | 3 |

Zusaetzlich enthaelt der Pilot-Tokenizer:

- `<|system|>`
- `<|user|>`
- `<|assistant|>`

## Lokale Pruefbefehle unter Windows PowerShell

```powershell
cd C:\LumenQuantum
.\.venv\Scripts\Activate.ps1
python scripts\inspect_model_size.py --config configs\quantum_1_base_pilot.yaml
python scripts\validate_quantum_model.py --config configs\quantum_1_base_pilot.yaml
python -m pytest tests\test_quantum_model.py
```

Ein optionaler Minimal-Trainingslauf zum Pruefen der Checkpoint-Struktur:

```powershell
python scripts\train_quantum_pilot.py --config configs\quantum_1_base_pilot.yaml --max-steps 1
```

Dieser Befehl erzeugt lokale Pilot-Artefakte unter `models/quantum-1-base/`.
Er ist kein grosses Training.
