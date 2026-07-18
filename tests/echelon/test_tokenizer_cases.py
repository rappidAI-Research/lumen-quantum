TEST_CASES = [
    "",
    " ",
    "\n",
    "\r\n",
    "\t",
    "Hallo Welt",
    "ä ö ü Ä Ö Ü ß",
    "„Deutsche Anführungszeichen“ und ‚einfache‘",
    "Berlin ist die Hauptstadt von Deutschland.",
    "Zeile 1\nZeile 2\nZeile 3",
    "Tabs:\tSpalte 1\tSpalte 2",
    "Emoji: 😀 🚀 ❤️ 🧠",
    "UTF-8: café, naïve, façade, İstanbul",
    '{"name":"Jonas","project":"quantum-1-echelon"}',
    "# Überschrift\n\n- Punkt 1\n- Punkt 2",
    "def add(a, b):\n    return a + b",
    "<|system|>\nDu bist ein hilfreicher Assistent.\n<|end|>",
    "<|user|>\nErkläre Quantenphysik einfach.\n<|end|>",
    "<|assistant|>\nQuantenphysik beschreibt...\n<|end|>",
    (
        "<|system|>\nDu bist hilfreich.\n<|end|>\n"
        "<|user|>\nHallo!\n<|end|>\n"
        "<|assistant|>\nHallo, wie kann ich helfen?\n<|end|>"
    ),
    "1234567890",
    "https://example.com/test?q=ä",
    "E-Mail: test@example.com",
]
