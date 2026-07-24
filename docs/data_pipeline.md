# rappidAI Quantum — quantum-1 Datenpipeline

> Historical pilot procedure from the earlier Lumen project phase. The current
> project identity is rappidAI Quantum; this document is retained for provenance.

Diese Pipeline bereitet eine kleine, reproduzierbare deutsche Pilotmenge aus `epfml/FineWeb2-HQ`, Subset `deu_Latn`, vor.

Es wird kein quantum-1-Modell trainiert, kein Tokenizer trainiert und es werden keine Modellgewichte geladen.

## Ausgabeordner

```text
data/quantum/raw/
data/quantum/cleaned/
data/quantum/manifests/
data/quantum/reports/
```

## Windows lokal

Aus dem Repository-Root:

```powershell
cd C:\path\to\rappidai-quantum
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\download_quantum_data.py --config configs\quantum_1_data.yaml
python scripts\clean_quantum_data.py --config configs\quantum_1_data.yaml
python scripts\sample_quantum_data.py --config configs\quantum_1_data.yaml
python scripts\inspect_quantum_data.py --config configs\quantum_1_data.yaml
python scripts\build_data_manifest.py --config configs\quantum_1_data.yaml
```

Der Download nutzt Streaming und stoppt bei maximal 100.000 Dokumenten oder 2 GB Rohtext, je nachdem was zuerst erreicht wird.

## Kleiner Pilotlauf

Fuer einen noch kleineren Test kannst du in `configs/quantum_1_data.yaml` temporaer setzen:

```yaml
download:
  max_documents: 1000
  max_raw_bytes: 50000000
```

Danach die fuenf Befehle erneut ausfuehren.

## RunPod spaeter

Auf RunPod ist die Reihenfolge gleich. Erst Repository und Umgebung vorbereiten:

```bash
cd /workspace/LumenQuantum
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Dann nur die Datenpipeline:

```bash
python scripts/download_quantum_data.py --config configs/quantum_1_data.yaml
python scripts/clean_quantum_data.py --config configs/quantum_1_data.yaml
python scripts/sample_quantum_data.py --config configs/quantum_1_data.yaml
python scripts/inspect_quantum_data.py --config configs/quantum_1_data.yaml
python scripts/build_data_manifest.py --config configs/quantum_1_data.yaml
```

Noch kein Training starten. Erst Manifest und Report pruefen.

## Splits

Die Splits werden nicht zufaellig nach dem Download gezogen. Jedes bereinigte Dokument bekommt einen stabilen SHA256-Hash. Daraus wird mit dem Split-Seed ein Bucket berechnet:

```text
sha256(f"{split_seed}:{document_sha256}")
```

Zielverteilung:

- Train: 98 %
- Validation: 1 %
- Test: 1 %

Bei kleinen Pilotmengen koennen Validation oder Test sehr klein sein. Fuer echte Bewertung braucht es spaeter deutlich groessere Mengen.

## Qualitaetsfilter

Die Cleaning-Stufe entfernt:

- leere Texte
- sehr kurze Texte
- kaputtes Unicode
- exakte Duplikate
- Texte mit extremen Zeichenwiederholungen
- Texte mit zu vielen Nicht-Buchstaben
- offensichtliches Boilerplate wie Cookie-, Newsletter-, Login- oder Warenkorb-Texte

Quellen-URL und Metadaten bleiben in separaten Feldern und werden nicht in den Trainingstext geschrieben.
