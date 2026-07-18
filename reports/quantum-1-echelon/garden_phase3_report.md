# quantum-1-echelon – Garden Phase 3

## Status

Die Garden-Datenpipeline wurde implementiert und durch Smoke-, Integritäts-,
Abbruch-, Resume- und Shutdown-Tests validiert.

Der vollständige Produktionslauf wurde noch nicht gestartet.

## Quelle

- Dataset: `epfml/FineWeb2-HQ`
- Konfiguration: `deu_Latn`
- Split: `train`
- Streaming: aktiviert
- Gepinnte Revision:
  `c0c06e94fd3a44ae9e802b2b0fc533817601eb5e`

## Tokenizer

- SentencePiece BPE
- Vokabulargröße: 32.768
- Datentyp der Shards: `uint16`
- EOS wird zwischen Dokumenten ergänzt
- Umlaute und Unicode-Roundtrip wurden validiert

## Qualitätsfilter

- Sprache: Deutsch
- Schriftsystem: Lateinisch
- Mindest-Sprachscore: 0,98
- Mindest-Quality-Score: 0,25
- Maximale MinHash-Clustergröße: 500
- Mindestlänge: 300 Zeichen
- Mindestwortzahl: 60
- Mindestanteil alphabetischer Zeichen: 0,55
- URL-, Ziffern-, Symbol-, Wiederholungs- und Boilerplate-Filter

FineWeb2 wurde bereits sprachweise dedupliziert. Ein zusätzlicher
produktionsweiter In-Memory-SimHash-Index ist daher deaktiviert.

## Finaler Smoke-Test

- Gesehene Dokumente: 5.001
- Akzeptierte Dokumente: 1.559
- Akzeptanzrate: 31,17 %
- Erzeugte Tokens: 1.380.886

## Produktionsziele

- Train: 8.000.000.000 Tokens
- Validation: 10.000.000 Tokens
- Test: 10.000.000 Tokens
- Shard-Größe: 100.000.000 Tokens
- Kontextlänge: 2.048 Tokens

## Ausfallsicherheit

Implementiert wurden:

- atomare JSON-Checkpoints
- sichere `.bin.part`-Dateien
- Rollback nicht checkpointeter Shard-Fortschritte
- Dataset-Resume über `state_dict`
- deterministischer `skip`-Fallback
- kontrollierter Teststopp
- sauberer Prozessabschluss trotz HF-Streaming-Shutdown-Problem

## Validierung

- 78 Tests bestanden
- 3 Tests übersprungen
- Mini-Produktionstest bestanden
- Shard-Integrität bestanden
- Crash-Resume-Test bestanden
- Dataset-State-Resume bestanden
- Unicode-/Umlaut-Roundtrip bestanden
