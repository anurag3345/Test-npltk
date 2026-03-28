from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


class LemmaDictionary:
    def __init__(self, dictionary_path: str | Path) -> None:
        self._path = Path(dictionary_path)
        self._entries = self._load(self._path)

    @staticmethod
    def _load(path: Path) -> Dict[str, str]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Lemma dictionary must be a JSON object")

        # Flat format: {"word": "lemma", ...}
        if all(isinstance(v, str) for v in data.values()):
            return {str(k): str(v) for k, v in data.items()}

        # Compact format: {"lemma": ["word1", "word2", ...], ...}
        if all(isinstance(v, list) for v in data.values()):
            entries: Dict[str, str] = {}
            for lemma, forms in data.items():
                lemma_s = str(lemma)
                entries[lemma_s] = lemma_s
                for form in forms:
                    if not isinstance(form, str):
                        raise ValueError("Compact lemma dictionary forms must be strings")
                    entries[form] = lemma_s
            return entries

        raise ValueError(
            "Lemma dictionary must be either flat {word: lemma} or compact {lemma: [forms]}"
        )

    def lookup(self, word: str) -> Optional[str]:
        return self._entries.get(word)
