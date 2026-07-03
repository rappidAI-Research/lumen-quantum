# Lumen Quantum Tokenizer-Spezifikation

Diese Spezifikation beschreibt den Pilot-Tokenizer `quantum-1-pilot`. Er ist
noch nicht der finale eingefrorene Tokenizer fuer `quantum-1`, sondern ein
kontrollierter Zwischenschritt fuer die echte Datenpipeline.

## Grundregeln

- Es werden keine vortrainierten Tokenizer verwendet.
- Es werden keine Modellgewichte geladen.
- Trainiert wird nur auf `data/quantum/cleaned/train.jsonl`.
- `validation.jsonl` und `test.jsonl` duerfen nicht in das Tokenizer-Training einfliessen.
- Der finale quantum-1-Tokenizer wird spaeter auf einer groesseren dokumentierten Datenmenge neu trainiert.

## Verfahren

- Algorithmus: SentencePiece BPE
- Pilot-Vokabular: 16384 Tokens
- Zielordner: `tokenizer/quantum-1-pilot/`
- HF-/GGUF-Klasse: `LlamaTokenizer`
- Byte-Fallback: aus
- Normalisierung: NFKC

## Sondertokens

Die klassischen LLaMA-IDs muessen stabil bleiben:

| Token | ID | Zweck |
| --- | ---: | --- |
| `<unk>` | 0 | unbekannte Zeichen/Stuecke |
| `<s>` | 1 | BOS |
| `</s>` | 2 | EOS |
| `<pad>` | 3 | Padding |

Zusaetzliche Chat-Tokens werden als SentencePiece `user_defined_symbols`
trainiert und liegen ebenfalls direkt in `tokenizer.model`:

- `<|system|>`
- `<|user|>`
- `<|assistant|>`

## Erzeugte Dateien

Im Ordner `tokenizer/quantum-1-pilot/` muessen mindestens diese Dateien liegen:

- `tokenizer.model`
- `tokenizer.vocab`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `tokenizer_manifest.json`

`tokenizer_manifest.json` dokumentiert Trainingsdatei, Hash der Trainingsdaten,
Vokabulargroesse, Token-IDs aller Sondertokens, SentencePiece-Version, Seed und
Erstellungsdatum.

## Validierung

`scripts/validate_quantum_tokenizer.py` prueft:

- deutsche Umlaute `ä`, `ö`, `ü`, `ß`
- Encode/decode-Roundtrip
- Sondertokens und ihre IDs
- keine UNK-Tokens in normalen deutschen Validierungssaetzen
- `vocab_size` gegen die spaetere LLaMA-Konfiguration
- strukturelle GGUF-Kompatibilitaet vor dem Export
