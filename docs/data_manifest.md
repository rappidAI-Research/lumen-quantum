# Data Manifest

Dieses Dokument beschreibt, welche Informationen fuer jede quantum-1-Datenversion dokumentiert werden muessen.

Die automatisch erzeugte Manifest-Datei liegt nach dem Pipeline-Lauf unter:

```text
data/quantum/manifests/data_manifest.md
data/quantum/manifests/data_manifest.json
```

## Pflichtfelder

- Dataset-Name und Version
- Erstellungszeitpunkt in UTC
- Seed
- verwendete YAML-Konfiguration inklusive SHA256
- Quellen und Lizenzen
- Filterregeln
- Anzahl Rohdokumente, bereinigte Dokumente und verworfene Dokumente
- Train-, Validation- und Test-Dokumentanzahlen
- ungefaehre Tokenzahlen ohne quantum-1-Tokenizer
- Datei-Hashes fuer Reproduzierbarkeit
- Bestaetigung, dass Splits per SHA256 disjunkt sind

## Aktueller Status

`quantum-1-data-smoke-v0` ist nur eine kleine technische Stichprobe. Sie ist nicht der finale Trainingsdatensatz fuer ein 50M-Parameter-Modell.

Vor echtem Training muessen Quellen, Lizenzlage, Datenmenge, Qualitaetsfilter und Evaluationsdaten erneut geprueft und versioniert werden.
