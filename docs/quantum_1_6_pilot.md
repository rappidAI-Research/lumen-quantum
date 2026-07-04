# quantum-1.6-pilot — Continued Pretraining

Diese Version setzt das Training von **quantum-1-pilot** fort (weights-only Continued
Pretraining). Gleiche Architektur, gleicher eingefrorener Tokenizer, gleiche
Kontextlaenge 512. Das Modell soll nach diesem Lauf insgesamt ca. **600 Mio. Tokens**
gesehen haben: ~100 Mio. aus quantum-1-pilot + ~500 Mio. neue deutsche Tokens.

> Vorbereitungsstatus: Code, Konfiguration, Tests und Doku sind fertig.
> Es wurde noch **kein** Daten-Preprocessing und **kein** Training gestartet.

## Wichtige Fakten und eine Pfad-Klarstellung

- **Basismodell (Gewichte):** `models/quantum-1-base/final` (enthaelt `model.safetensors`).
- **Eingefrorener Tokenizer:** real unter **`tokenizer/quantum-1-pilot/`**.
  Die urspruengliche Aufgabenstellung nannte `tokenizer/quantum-1`; dieser Pfad
  existiert im Repo nicht. Der Tokenizer, den quantum-1-pilot tatsaechlich benutzt
  hat, liegt unter `tokenizer/quantum-1-pilot/` und ist **byte-identisch** zum
  Tokenizer im Basismodell (`models/quantum-1-base/final/tokenizer.model`,
  SHA256 gleich). Der Preflight erzwingt diese Identitaet.
- **Parameterzahl:** exakt **49.295.872** (16384·512 tied + 12·3.408.896 + 512).
- **Der Tokenizer wird nie neu trainiert oder veraendert.**
- **quantum-1-pilot, seine Daten und seine Evaluation bleiben unveraendert.**

## Neue Dateien

| Datei | Zweck |
|---|---|
| `configs/quantum_1_6_pilot_data.yaml` | Datenpipeline (Download → Clean → Overlap-Filter → Split → Tokenisierung → Manifest) |
| `configs/quantum_1_6_pilot_train.yaml` | Trainingskonfiguration (weights-only Init, LR 1e-4, 30.518 Schritte) |
| `scripts/quantum_1_6_preflight.py` | Torch-freie Preflight-Checks (Config, Tokenizer-Hash, Pfad-Isolation) |
| `scripts/exclude_known_documents.py` | Entfernt Ueberschneidungen mit dem alten Datensatz |
| `scripts/train_quantum_continued.py` | Weights-only Continued Pretraining |
| `tests/test_quantum_1_6_pilot.py` | Tests (Config, Tokenizer, Overlap, weights-only Init, Checkpoints) |
| `docs/quantum_1_6_pilot.md` | Dieses Dokument |

Bestehende Skripte (`download_quantum_data.py`, `clean_quantum_data.py`,
`sample_quantum_data.py`, `tokenize_quantum_data.py`, `build_data_manifest.py`)
werden **unveraendert** wiederverwendet und nur ueber die neue Config gesteuert.

## Wie die Datenueberschneidung behandelt wird

1. **Anderer Seed** (`20260716`) als quantum-1-pilot (`20260703`): der Streaming-Shuffle
   von FineWeb2-HQ liefert eine andere Dokumentreihenfolge.
2. **Fingerprint-Ausschluss** (`scripts/exclude_known_documents.py`): baut aus dem alten
   Datensatz (`data/quantum/cleaned/*.jsonl`) eine Menge aus normalisiertem
   Text-SHA256 (`sha256`) und stabiler Dokument-ID (`id`) und entfernt jedes neue
   Dokument, dessen Fingerprint bereits im alten Satz vorkommt.
3. **Report:** `data/quantum/quantum_1_6_pilot/reports/overlap_report.json` mit
   Anzahl entfernter Dokumente und Overlap-Quote.

**Bekannte Grenze:** Es werden nur exakte Fingerprints verglichen. Near-Duplicates
(leichte Umformulierungen, andere Whitespace-/HTML-Varianten) werden nicht erkannt.
Ohne Volldeduplikation ueber das gesamte Korpus ist keine perfekte Disjunktheit
garantiert. Der alte Datensatz wird dabei ausschliesslich gelesen, nie veraendert.

## Wie quantum-1.6-pilot aus quantum-1-pilot initialisiert wird

`scripts/train_quantum_continued.py`:

1. Baut das Modell frisch aus `LlamaConfig` (gleiche Dimensionen).
2. Prueft die Parameterzahl **exakt** (`== 49.295.872`).
3. Laedt **nur die Gewichte** aus `models/quantum-1-base/final` via
   `load_quantum_weights` (kein `from_pretrained`, keine Neuinitialisierung) und
   verifiziert anhand eines Embedding-Fingerprints, dass die Gewichte sich
   tatsaechlich veraendert haben (also wirklich geladen wurden).
4. Uebernimmt **ausdruecklich nicht**: alten Optimizer-, Scheduler- oder
   Schrittzustand. Es werden ein **frischer AdamW-Optimizer** und ein **frischer
   Cosine-Scheduler** erstellt, `global_step = 0`.
5. LR 1e-4, Warmup 500 Schritte, Cosine-Decay, `max_grad_norm = 1.0`.
6. Checkpoints und Evaluation mindestens alle 1.000 Schritte.
7. Feste deutsche Beispiel-Prompts werden bei jeder Evaluation generiert und nach
   `logs/quantum_1_6_pilot/sample_generations.jsonl` geschrieben (vergleichbar ueber
   die Zeit).
8. Output ausschliesslich nach `models/quantum-1.6-pilot/` (per Assertion erzwungen).

`--resume-from auto` setzt einen **eigenen** 1.6-Lauf fort (dann inkl.
Optimizer/Scheduler/Schritt aus dem 1.6-Checkpoint) — nicht den alten Basislauf.

## Effektive Batchgroesse und Schrittzahl

`batch_size 8 × grad_accum 4 × 512 Tokens = 16.384 Tokens/Schritt` (Single-GPU).
`500.000.000 / 16.384 ≈ 30.518 Schritte`.

Bei knapperem VRAM die gleiche effektive Batchgroesse halten, z. B.
`batch_size 4 × grad_accum 8` oder `batch_size 2 × grad_accum 16`.

---

## Exakte RunPod-Befehle

### 0. Einrichtung (einmalig pro Instanz)

```bash
cd /workspace/LumenQuantum
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Diese Schritte setzen voraus, dass `models/quantum-1-base/final/` und
`tokenizer/quantum-1-pilot/` auf der Instanz vorhanden sind (mit-synchronisieren).

### 1. Datenaufbereitung (~500 Mio. neue Train-Tokens)

```bash
# 1a. Streaming-Download aus FineWeb2-HQ deu_Latn (anderer Seed -> andere Reihenfolge)
python scripts/download_quantum_data.py --config configs/quantum_1_6_pilot_data.yaml

# 1b. Bereinigung (Boilerplate, Sprachqualitaet, zu kurze/kaputte Dokumente, Exakt-Dedup)
python scripts/clean_quantum_data.py --config configs/quantum_1_6_pilot_data.yaml

# 1c. Ueberschneidungen mit dem alten quantum-1-pilot-Datensatz entfernen
python scripts/exclude_known_documents.py --config configs/quantum_1_6_pilot_data.yaml

# 1d. Stabile, disjunkte Train/Validation/Test-Splits (SHA256-Buckets)
python scripts/sample_quantum_data.py --config configs/quantum_1_6_pilot_data.yaml

# 1e. Tokenisierung mit dem eingefrorenen Tokenizer; Train hart auf 500 Mio. Tokens begrenzt
python scripts/tokenize_quantum_data.py --config configs/quantum_1_6_pilot_data.yaml

# 1f. Ausfuehrliches Datenmanifest (Quelle, Revision, Seeds, Zaehlungen, Hashes, Filterregeln)
python scripts/build_data_manifest.py --config configs/quantum_1_6_pilot_data.yaml
```

Exakte Tokenzahlen danach aus
`data/quantum/quantum_1_6_pilot/tokenized/tokenization_manifest.json` ablesen.

> Speicherhinweis: `download_quantum_data.py` sammelt Dokumente im RAM. Fuer 500 Mio.
> Tokens ist eine Instanz mit viel RAM noetig; alternativ `download.max_raw_bytes` in
> der Data-Config reduzieren und in mehreren Shards arbeiten.

### 2. Preflight und Smoke-Test (kein langes Training)

```bash
# 2a. Torch-freie Sicherheitspruefungen (Config, Tokenizer-Identitaet, Pfad-Isolation)
python scripts/quantum_1_6_preflight.py \
  --config configs/quantum_1_6_pilot_train.yaml \
  --data-config configs/quantum_1_6_pilot_data.yaml --json

# 2b. Parameterzahl exakt pruefen (muss 49.295.872 sein)
python scripts/inspect_model_size.py --config configs/quantum_1_6_pilot_train.yaml --json

# 2c. Dry-Run: weights-only Init aus dem Basismodell + ein Forward Pass, nichts gespeichert
python scripts/train_quantum_continued.py --config configs/quantum_1_6_pilot_train.yaml --dry-run

# 2d. Kurzer Smoke-Test mit echten Daten (schreibt nach models/quantum-1.6-pilot/)
python scripts/train_quantum_continued.py --config configs/quantum_1_6_pilot_train.yaml --max-steps 5

# 2e. Tests
python -m pytest tests/test_quantum_1_6_pilot.py -v
```

### 3. Volltraining (~30.518 Schritte)

```bash
# Im Hintergrund/mit Logfile empfohlen (z. B. tmux):
python scripts/train_quantum_continued.py --config configs/quantum_1_6_pilot_train.yaml

# Fortsetzen nach Unterbrechung (eigener 1.6-Lauf):
python scripts/train_quantum_continued.py --config configs/quantum_1_6_pilot_train.yaml --resume-from auto
```

Artefakte:
- Checkpoints: `models/quantum-1.6-pilot/checkpoints/checkpoint-step-XXXXX/`
- Finales Modell: `models/quantum-1.6-pilot/final/`
- Trainingslog: `models/quantum-1.6-pilot/train.log`
- Beispiel-Generierungen ueber die Zeit: `logs/quantum_1_6_pilot/sample_generations.jsonl`

## Was unveraendert bleibt

Kein Eingriff in Training, Tokenizer oder Gewichte von quantum-1-pilot; keine
Aenderung an `tokenizer/quantum-1-pilot`, an der bestehenden Evaluation, an der
Android-App oder an GGUF-Dateien. Alle neuen Artefakte liegen unter
`data/quantum/quantum_1_6_pilot/`, `models/quantum-1.6-pilot/` und
`logs/quantum_1_6_pilot/`.
