"""Erzeugt synthetische deutsche Übungssätze für den technischen Smoke-Test.

Die Datei data/raw/smoke.txt dient ausschließlich dazu, Tokenizer,
Datenaufbereitung, Training und Textgenerierung technisch zu prüfen.

Merkmale:
- rein synthetisch aus Vorlagen und Wortlisten kombiniert
- keine echten personenbezogenen Daten (nur generische Rollen)
- keine kopierten Bücher, Songtexte oder Webseiten
- echte deutsche Umlaute und ß, damit die SentencePiece-Tokenisierung geprüft wird
- deterministisch über einen festen Seed (reproduzierbar)

Jeder Satz wird als eigener Absatz (durch Leerzeile getrennt) geschrieben,
sodass prepare_data.py ihn als eigene Texteinheit behandelt.
"""

from __future__ import annotations

import argparse
import random
from itertools import product
from pathlib import Path
from typing import Iterable

SEED = 42
PER_THEME = 1150
TARGET_MAX = 9000
TARGET_MIN = 6000

# Zeitangaben als vorangestellte Adverbiale (Verb-Zweitstellung: "Am Morgen trinke ich ...").
ZEIT = [
    "Am Morgen", "Am Vormittag", "Am Mittag", "Am Nachmittag", "Am Abend",
    "Heute", "Morgen", "Danach", "Am Wochenende", "Am Montag",
    "Am Freitag", "Jeden Tag", "Oft", "Manchmal", "Zwischendurch",
    "Im Sommer", "Im Winter", "Im Frühling", "Nach der Pause", "Vor dem Essen",
]


def transitive_sentences(verb_subject: list[str], objects: list[str]) -> list[str]:
    """Baut Sätze 'Zeit + Verb + Subjekt + Akkusativobjekt.'"""
    out = []
    for zeit, vs, obj in product(ZEIT, verb_subject, objects):
        out.append(f"{zeit} {vs} {obj}.")
    return out


def svo_sentences(subjects: list[str], predicates: list[str], suffixes: list[str]) -> list[str]:
    """Baut Sätze 'Subjekt + Prädikat (+ Zusatz).'"""
    out = []
    for subj, pred, suffix in product(subjects, predicates, suffixes):
        tail = f" {suffix}" if suffix else ""
        out.append(f"{subj} {pred}{tail}.")
    return out


def build_alltag() -> list[str]:
    vs = [
        "trinke ich", "koche ich", "kaufe ich", "esse ich", "hole ich",
        "bringe ich", "wasche ich", "suche ich", "trage ich", "packe ich",
        "brauche ich", "nehme ich", "kauft die Familie", "kocht der Vater",
        "holt der Nachbar", "trägt das Kind", "macht die Mutter", "bringt der Freund",
    ]
    obj = [
        "frisches Brot", "reifes Obst", "kalte Milch", "einen warmen Tee",
        "eine kleine Suppe", "frisches Gemüse", "einen roten Apfel", "saubere Wäsche",
        "eine leichte Tasche", "ein sauberes Handtuch", "frische Eier", "einen süßen Kuchen",
        "eine reife Banane", "kühles Wasser", "warme Brötchen",
    ]
    extra = [
        "Am Abend räume ich die Küche in Ruhe auf.",
        "Vor dem Schlafen lese ich noch eine Seite.",
        "Am Wochenende gehen wir gern im Park spazieren.",
        "Nach dem Essen spüle ich das Geschirr ab.",
        "Am Morgen öffne ich zuerst das Fenster.",
        "Der Bus fährt jede halbe Stunde.",
        "Wir treffen uns später am kleinen Platz.",
        "Das Kind wäscht sich vor dem Essen die Hände.",
    ]
    return transitive_sentences(vs, obj) + extra


def build_technik() -> list[str]:
    vs = [
        "repariere ich", "prüfe ich", "baue ich", "teste ich", "schalte ich",
        "messe ich", "montiere ich", "öle ich", "kontrolliere ich",
        "startet der Techniker", "prüft die Meisterin", "repariert der Nachbar",
        "baut das Team", "wartet der Fachmann",
    ]
    obj = [
        "die alte Maschine", "den kleinen Motor", "das lose Kabel", "die neue Lampe",
        "den runden Schalter", "das schwere Werkzeug", "die feine Schraube",
        "den langen Draht", "die stabile Leiter", "das kleine Ventil",
        "die laute Pumpe", "den warmen Ofen", "die schnelle Bohrmaschine",
    ]
    extra = [
        "Ein guter Motor läuft ruhig und gleichmäßig.",
        "Der Techniker zieht die Schraube vorsichtig fest.",
        "Ein Werkzeug gehört nach der Arbeit zurück in den Kasten.",
        "Die Maschine wird vor dem Start kurz geprüft.",
        "Ein loses Kabel kann eine Störung verursachen.",
        "Der Schalter klickt leise, wenn man ihn drückt.",
    ]
    return transitive_sentences(vs, obj) + extra


def build_schule() -> list[str]:
    vs = [
        "lerne ich", "lese ich", "übe ich", "schreibe ich", "erkläre ich",
        "wiederhole ich", "rechne ich", "male ich", "zeichne ich",
        "erklärt die Lehrerin", "liest der Schüler", "übt die Klasse",
        "korrigiert der Lehrer", "bearbeitet das Kind",
    ]
    obj = [
        "die neue Aufgabe", "eine kurze Frage", "das schwere Wort", "die lange Zahl",
        "den ganzen Satz", "die richtige Lösung", "das kleine Gedicht", "die bunte Karte",
        "die erste Seite", "das neue Thema", "die einfache Regel", "den kurzen Text",
    ]
    extra = [
        "In der Pause spielen die Kinder auf dem Hof.",
        "Die Lehrerin erklärt die Aufgabe noch einmal in Ruhe.",
        "Wer übt, macht mit der Zeit weniger Fehler.",
        "Frage: Wie viel ist drei plus vier? Antwort: Sieben.",
        "Die Klasse liest heute eine kurze Geschichte.",
        "Am Ende der Stunde räumen alle ihre Sachen auf.",
    ]
    return transitive_sentences(vs, obj) + extra


def build_computer() -> list[str]:
    vs = [
        "öffne ich", "speichere ich", "schließe ich", "kopiere ich", "lösche ich",
        "starte ich", "installiere ich", "aktualisiere ich", "durchsuche ich",
        "öffnet der Nutzer", "speichert das Programm", "lädt der Rechner",
        "zeigt der Bildschirm", "sichert das System",
    ]
    obj = [
        "die kleine Datei", "das lange Dokument", "den neuen Ordner", "das schnelle Programm",
        "die große Tabelle", "das scharfe Bild", "die kurze Notiz", "den langen Text",
        "die sichere Kopie", "das offene Fenster", "die lokale Datenbank", "den freien Speicher",
    ]
    extra = [
        "Ein Programm besteht aus vielen kleinen Befehlen.",
        "Der Rechner speichert die Datei auf der Festplatte.",
        "Ein Ordner kann viele Dateien enthalten.",
        "Vor dem Ausschalten sollte man die Arbeit speichern.",
        "Ein Backup schützt vor verlorenen Daten.",
        "Der Bildschirm zeigt das Ergebnis in Sekunden.",
        "Frage: Was macht ein Editor? Antwort: Er bearbeitet Text.",
    ]
    return transitive_sentences(vs, obj) + extra


def build_natur() -> list[str]:
    subj = [
        "Der Baum", "Die Blume", "Der Fluss", "Der Berg", "Die Wolke",
        "Der Wald", "Das Gras", "Die Biene", "Der Vogel", "Der Regen",
        "Die Sonne", "Der Wind", "Der See", "Das Blatt", "Der Schmetterling",
        "Der Stein", "Die Wiese",
    ]
    pred = [
        "ist heute gut zu sehen", "gehört zur Natur", "wirkt sehr ruhig",
        "gefällt vielen Menschen", "verändert sich mit der Zeit",
        "ist ein Teil der Landschaft", "fällt sofort auf",
        "bleibt lange in Erinnerung", "passt gut in die Umgebung",
        "zeigt sich im Licht", "ruht in der Stille", "liegt ruhig vor uns",
    ]
    suffix = ["", "im Sommer", "am Morgen", "in der Natur", "bei gutem Wetter", "meistens"]
    extra = [
        "Nach dem Regen riecht die Luft besonders frisch.",
        "Im Wald ist es kühl und still.",
        "Die Blätter färben sich im Herbst bunt.",
        "Bienen sammeln Nektar von vielen Blüten.",
        "Ein kleiner Bach fließt leise durch das Tal.",
    ]
    return svo_sentences(subj, pred, suffix) + extra


def build_wissenschaft() -> list[str]:
    subj = [
        "Wasser", "Eis", "Licht", "Luft", "Ein Magnet", "Die Schwerkraft",
        "Eine Pflanze", "Der Mond", "Die Sonne", "Ein Atom", "Der Schall",
        "Die Wärme", "Ein Kreis", "Eine Zahl", "Der Sauerstoff", "Ein Kristall",
    ]
    pred = [
        "lässt sich gut beobachten", "folgt einfachen Regeln", "kommt in der Natur vor",
        "ist ein Thema im Unterricht", "lässt sich einfach erklären",
        "spielt in Experimenten eine Rolle", "ist leicht zu messen",
        "gehört zur Physik", "wird oft untersucht", "zeigt ein klares Muster",
        "ist gut erforscht", "hat feste Eigenschaften",
    ]
    suffix = ["", "im Versuch", "im Alltag", "in der Schule", "meistens", "oft"]
    extra = [
        "Wasser kocht bei hundert Grad Celsius.",
        "Eis schwimmt auf flüssigem Wasser.",
        "Licht ist schneller als Schall.",
        "Pflanzen brauchen Sonne, Wasser und Luft.",
        "Der Mond umkreist die Erde.",
        "Warme Luft steigt nach oben.",
        "Ein Magnet zieht Eisen an.",
        "Salz löst sich gut in Wasser.",
        "Die Erde dreht sich um die Sonne.",
        "Schall braucht ein Medium, um sich auszubreiten.",
    ]
    return svo_sentences(subj, pred, suffix) + extra


def build_lumen() -> list[str]:
    vs = [
        "öffnet Lumen", "speichert Lumen", "zeigt Lumen", "liest Lumen",
        "schreibt Lumen", "sucht Lumen", "erklärt Lumen", "prüft Lumen",
        "sortiert Lumen", "beantwortet Lumen", "findet Lumen", "ergänzt Lumen",
    ]
    obj = [
        "die passende Datei", "eine kurze Antwort", "den ganzen Text", "die richtige Notiz",
        "eine einfache Frage", "den lokalen Ordner", "die kleine Tabelle", "ein neues Beispiel",
        "die letzte Zeile", "eine klare Erklärung", "den nächsten Schritt", "die gespeicherte Liste",
    ]
    imperativ = [
        "öffne die letzte Datei", "speichere den Text", "zeige die Liste",
        "lies die Notiz vor", "erkläre den Satz einfach", "fasse das Kapitel kurz zusammen",
        "suche das Wort im Text", "sortiere die Zeilen", "prüfe die Aufgabe",
        "wiederhole die letzte Antwort",
    ]
    extra = [
        "Lumen ist ein lokaler, deutschsprachiger Assistent.",
        "Lumen läuft offline auf dem eigenen Gerät.",
        "Lumen antwortet ruhig, klar und hilfreich.",
        "Lumen speichert nichts ohne deine Erlaubnis.",
        "Lumen arbeitet ohne Verbindung zum Internet.",
        "Frage: Was ist Lumen? Antwort: Lumen ist ein lokaler Testassistent.",
        "Frage: Läuft Lumen offline? Antwort: Ja, Lumen läuft lokal.",
        "Frage: Was kann Lumen? Antwort: Lumen liest, schreibt und erklärt Texte.",
        "Lumen hilft beim Ordnen von Notizen und Dateien.",
        "Lumen gibt kurze und freundliche Antworten.",
    ]
    sentences = transitive_sentences(vs, obj)
    sentences += [f"Lumen, {imp}." for imp in imperativ]
    sentences += extra
    return sentences


THEMES = {
    "alltag": build_alltag,
    "technik": build_technik,
    "schule": build_schule,
    "natur": build_natur,
    "wissenschaft": build_wissenschaft,
    "computer": build_computer,
    "lumen": build_lumen,
}


def generate(seed: int, per_theme: int) -> list[str]:
    rng = random.Random(seed)
    collected: list[str] = []
    seen: set[str] = set()
    for _name, builder in THEMES.items():
        pool = list(dict.fromkeys(builder()))  # exakte Duplikate je Thema entfernen
        rng.shuffle(pool)
        for sentence in pool[:per_theme]:
            if sentence not in seen:
                seen.add(sentence)
                collected.append(sentence)
    rng.shuffle(collected)
    return collected


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Erzeugt synthetische Smoke-Test-Daten.")
    parser.add_argument("--output", default="data/raw/smoke.txt", help="Zieldatei.")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed fuer Reproduzierbarkeit.")
    parser.add_argument("--per-theme", type=int, default=PER_THEME, help="Maximale Saetze pro Thema.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    sentences = generate(args.seed, args.per_theme)
    if len(sentences) > TARGET_MAX:
        sentences = sentences[:TARGET_MAX]
    if len(sentences) < TARGET_MIN:
        raise SystemExit(
            f"Nur {len(sentences)} Saetze erzeugt (< {TARGET_MIN}). Wortlisten erweitern."
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(sentences) + "\n", encoding="utf-8")
    print(f"{len(sentences)} Saetze -> {output}")


if __name__ == "__main__":
    main()
