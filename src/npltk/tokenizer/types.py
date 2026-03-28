from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TokenType(str, Enum):
    WORD_DEV = "WORD_DEV"       # Devanagari full word (rule-based path)
    SUBWORD_DEV = "SUBWORD_DEV" # Devanagari sub-word piece (BPE model path)
    WORD_LAT = "WORD_LAT"       # Latin word (code-mix)
    NUM = "NUM"                 # numbers, dates, times
    PUNCT = "PUNCT"             # punctuation
    EMOJI = "EMOJI"             # emoji-like characters (heuristic)
    SYMBOL = "SYMBOL"           # other symbols / unknown


@dataclass(frozen=True)
class Token:
    text: str
    start: int  # start offset in the input string
    end: int    # end offset (exclusive)
    type: TokenType
