# Project Spec: Lumen Quantum

## Ziel

Lumen ist der Name des spaeteren deutschsprachigen Assistenten. Dieses Repository dient dazu, die komplette Trainingspipeline fuer ein eigenes Decoder-only-Sprachmodell von Grund auf aufzubauen.

## Modellfamilie

- `smoke-5m`: sehr kleines Smoke-Test-Modell mit ungefaehr 5 Millionen Parametern.
- `quantum-1`: spaeteres echtes Modell mit etwa 50 Millionen Parametern.

`quantum-1` wird erst gebaut, wenn der Smoke-Test vollstaendig funktioniert.

## Harte Regel: Keine vortrainierten Modellgewichte

Dieses Projekt darf niemals vortrainierte Modellgewichte laden.

- Kein `from_pretrained` fuer Modelle.
- Keine externen Modellgewichte.
- Keine API-Modelle.
- Keine versteckten Initialisierungen aus bestehenden Checkpoints, ausser lokalen Checkpoints, die mit diesem Projekt selbst erzeugt wurden.

Das Modell muss mit zufaelligen Gewichten starten. Lokale Checkpoints duerfen nur zum Fortsetzen eigener Trainingslaeufe genutzt werden.

## Reproduzierbarkeit

Jede Trainingsversion muss reproduzierbar dokumentiert werden. Dazu gehoeren:

- YAML-Konfiguration
- Modellarchitektur
- Tokenizer-Version und Tokenizer-Metadaten
- Seed
- Datensatzversion oder Rohdatenstand
- Train/Validation/Test-Split
- Kontextlaenge
- Trainingsparameter
- Checkpoint-Pfade
- Datum und Zweck des Trainingslaufs

Die Smoke-Version schreibt Metadaten in:

- `tokenizer/smoke/tokenizer_metadata.json`
- `data/processed/split_metadata.json`
- `data/tokenized/metadata.json`
- `models/smoke/final/training_metadata.json`

## Smoke-Test-Pipeline

Der erste funktionsfaehige Ablauf ist:

```text
Textdatei in data/raw/
-> Tokenizer trainieren
-> Daten vorbereiten und tokenisieren
-> kleines Modell trainieren
-> Checkpoint speichern
-> Training fortsetzen
-> Text im Terminal generieren
-> einfache Evaluation durchfuehren
```

## Technische Basis

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Tokenizers
- Accelerate
- YAML-Konfiguration
- Logging
- Checkpoints mit Resume-Funktion
- CPU- und CUDA-Unterstuetzung

## Smoke-Modell

- Architektur: `LlamaForCausalLM`
- Konfiguration: `LlamaConfig`
- Initialisierung: zufaellig
- Kontextlaenge: 256 Tokens
- Tokenizer: eigener BPE-Tokenizer
- Sprache: Deutsch
- Zweck: Pipeline pruefen, nicht Qualitaet maximieren

## Naechster Meilenstein

Erst wenn der Smoke-Test stabil laeuft, wird `quantum-1` entworfen. Dann muessen Datensatz, Tokenizer, Modellgroesse, Trainingsdauer, Evaluationsverfahren und Checkpoint-Strategie neu versioniert und dokumentiert werden.
