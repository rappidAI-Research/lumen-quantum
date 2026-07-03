# Data Manifest

Dieses Dokument beschreibt die Pflichtangaben fuer jede quantum-1-Datenversion.

Die automatisch erzeugten Manifest-Dateien liegen nach dem Pipeline-Lauf unter:

```text
data/quantum/manifests/data_manifest.json
data/quantum/manifests/data_manifest.md
```

## Pflichtfelder

- Dataset-Name und Version
- Quelle: `epfml/FineWeb2-HQ`, Subset `deu_Latn`
- Dataset-Revision
- Download-Zeitpunkt in UTC
- Lizenzhinweis aus der Dataset Card
- Seed fuer Streaming-Shuffle
- Split-Seed fuer stabile Hash-Buckets
- Filterregeln
- Dokumentanzahlen fuer Raw, Cleaned, Train, Validation und Test
- Textmenge in Zeichen und Woertern
- geschaetzte Tokenanzahl ohne quantum-1-Tokenizer
- SHA256-Hashes der erzeugten Dateien
- Split-Methode und Disjunktheitskriterium

## Aktueller Status

`quantum-1-fineweb2hq-deu-pilot-v0` ist eine Pilotmenge. Sie dient zum Testen der echten Datenpipeline und ist noch kein finaler Trainingsdatensatz fuer ein 50M-Parameter-Modell.

Vor echtem Training muessen Quelle, Lizenzlage, Datenmenge, Filterregeln, Tokenizer und Evaluationsdaten erneut geprueft und versioniert werden.
