from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any

from npltk.tokenizer.types import Token, TokenType


class StopWordRemover:
    def __init__(self, stopword_file: str | None = None):
        if stopword_file is None:
            stopword_file = Path(__file__).parent / "nepali_stopwords.txt"

        self.stopwords = self._load_stopwords(stopword_file)

        # Your enum includes WORD_DEV, so treat any TokenType starting with "WORD" as word-token
        self.word_like_types = {t for t in TokenType if t.name.startswith("WORD")}

    def _load_stopwords(self, path: str) -> set[str]:
        with open(path, "r", encoding="utf-8") as f:
            return {
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            }

    def _tok_text(self, tok: Token) -> str | None:
        """
        Your Token prints as Token(text='...', ...), so prefer .text.
        Fallback to .value if present.
        """
        val = getattr(tok, "text", None)
        if isinstance(val, str):
            return val
        val = getattr(tok, "value", None)
        if isinstance(val, str):
            return val
        return None

    def remove(self, tokens: List[Token]) -> Tuple[List[Token], Dict[str, Any]]:
        filtered: List[Token] = []
        removed_words: List[str] = []

        for tok in tokens:
            val = self._tok_text(tok)
            ttype = getattr(tok, "type", None)

            if (
                isinstance(val, str)
                and (not self.word_like_types or ttype in self.word_like_types)
                and val in self.stopwords
            ):
                removed_words.append(val)
                continue

            filtered.append(tok)

        return filtered, {
            "removed_words": removed_words,
            "removed_count": len(removed_words),
            "changed": len(removed_words) > 0,
        }