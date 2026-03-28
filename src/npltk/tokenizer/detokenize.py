from __future__ import annotations

from typing import List

from .types import Token, TokenType


_OPENING_PUNCT = ("(", "[", "{", "\u00ab", "\u201c", "\u2018")


def detokenize_tokens(tokens: List[Token]) -> str:
    """
    Reconstruct text from tokens.

    Sub-word pieces are joined directly (no space before SUBWORD_DEV unless
    it is the first token or follows punctuation).
    """
    parts: List[str] = []

    for i, tok in enumerate(tokens):
        if i == 0:
            parts.append(tok.text)
        elif tok.type == TokenType.SUBWORD_DEV:
            parts.append(tok.text)
        elif tok.type == TokenType.PUNCT:
            parts.append(tok.text)
        elif tokens[i - 1].type == TokenType.PUNCT and tokens[i - 1].text in _OPENING_PUNCT:
            parts.append(tok.text)
        else:
            parts.append(" " + tok.text)

    return "".join(parts)
