"""
rule_engine.py
--------------
Stage-1 of the hybrid pipeline.

Converts raw (normalized) text into coarse PreToken chunks using
fast regex rules. Only WORD_DEV chunks are forwarded to the
SentencePiece model in stage-2; all other types are finalized here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .types import Token, TokenType


# ---------------------------------------------------------------------------
# Unicode ranges
# ---------------------------------------------------------------------------
DEV_RANGE = r"\u0900-\u0963\u0966-\u097F"   # Devanagari block
NEP_DIGIT = r"\u0966-\u096F"                 # Nepali digits ०–९

PUNCT_CHARS = r"""।!?.,;:…—–\-(){}[\]<>«»""'''/\\|@#%^&*_+=~`"""

# ---------------------------------------------------------------------------
# Named-group master regex  (order matters!)
# ---------------------------------------------------------------------------
def _build_pattern() -> re.Pattern:
    url     = r"(?P<URL>(https?://|www\.)\S+)"
    number  = rf"(?P<NUMBER>[\d{NEP_DIGIT}]+(?:[,\.:/\-][\d{NEP_DIGIT}]+)*)"
    dev     = rf"(?P<WORD_DEV>[{DEV_RANGE}]+)"
    lat     = r"(?P<WORD_LAT>[A-Za-z]+(?:'[A-Za-z]+)?)"
    punct   = rf"(?P<PUNCT>[{re.escape(PUNCT_CHARS)}])"
    ws      = r"(?P<WS>\s+)"
    other   = r"(?P<OTHER>.)"
    return re.compile("|".join([url, number, dev, lat, punct, ws, other]))


_MASTER = _build_pattern()


def _is_emoji(ch: str) -> bool:
    code = ord(ch)
    return (
        0x1F300 <= code <= 0x1FAFF
        or 0x2600 <= code <= 0x26FF
        or 0x2700 <= code <= 0x27BF
    )


# ---------------------------------------------------------------------------
# PreToken — output of rule engine, input to model router
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PreToken:
    text: str
    start: int
    end: int
    type: TokenType


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def pre_tokenize(text: str, keep_punct: bool = True) -> List[PreToken]:
    """
    Segment *text* into coarse PreToken chunks.

    WORD_DEV chunks → forwarded to SentencePiece model.
    Everything else → converted directly to final Token by the caller.
    """
    tokens: List[PreToken] = []

    for m in _MASTER.finditer(text):
        kind = m.lastgroup
        val  = m.group(0)
        s, e = m.start(), m.end()

        if kind == "WS":
            continue
        elif kind == "URL":
            tokens.append(PreToken(val, s, e, TokenType.SYMBOL))
        elif kind == "NUMBER":
            tokens.append(PreToken(val, s, e, TokenType.NUM))
        elif kind == "WORD_DEV":
            tokens.append(PreToken(val, s, e, TokenType.WORD_DEV))
        elif kind == "WORD_LAT":
            tokens.append(PreToken(val, s, e, TokenType.WORD_LAT))
        elif kind == "PUNCT":
            if keep_punct:
                tokens.append(PreToken(val, s, e, TokenType.PUNCT))
        else:  # OTHER
            tp = TokenType.EMOJI if (len(val) == 1 and _is_emoji(val)) else TokenType.SYMBOL
            tokens.append(PreToken(val, s, e, tp))

    return tokens
