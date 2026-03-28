"""
word_tokenizer.py
-----------------
Thin wrapper around the rule-engine pre-tokenizer.
Converts PreTokens → Tokens for the public API.
"""
from __future__ import annotations

from typing import List

from .rule_engine import pre_tokenize
from .types import Token


def tokenize_words(text: str, *, keep_punct: bool = True) -> List[Token]:
    """
    Rule-based word tokenizer.
    Works on normalized text and returns tokens with spans and types.
    """
    return [
        Token(pt.text, pt.start, pt.end, pt.type)
        for pt in pre_tokenize(text, keep_punct=keep_punct)
    ]
