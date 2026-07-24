# rappidAI Quantum — quantum-1-echelon Garden pipeline

> Current strategic-line planning document. Production-data preparation and
> Echelon training remain incomplete unless a signed run manifest says otherwise.

## Ziel
Reproduzierbare Vorbereitung hochwertiger Trainingsdaten für
quantum-1-echelon-base.

## Tokenbudget
- Mindestziel: 5 Milliarden Tokens
- Bevorzugtes Ziel: 8 bis 10 Milliarden Tokens

## Regeln
- Keine Quantum-1- oder Quantum-1.6-Daten wiederverwenden
- Quellen, Revisionen und Lizenzen dokumentieren
- Sprach- und Qualitätsfilter anwenden
- Boilerplate, URL-Müll und Zahlenketten entfernen
- Exakte und nahe Duplikate entfernen
- Wiederholte N-Gramme prüfen
- Stabile Train-, Validation- und Test-Splits verwenden
- Erst danach mit dem neuen Echelon-Tokenizer tokenisieren

## Speicher
Der Pod hat etwa 300 GB Speicher.
Zwischenstände erst nach erfolgreicher Prüfung von Manifesten und Hashes löschen.
