"""
tokenizer.py
------------
Public-facing rule-based tokenizer for Nepali text.
"""
from __future__ import annotations

from typing import List

from .detokenize import detokenize_tokens
from .sentence_splitter import split_sentences
from .types import Token, TokenizedSentence
from .word_tokenizer import tokenize_words


class NepaliTokenizer:
    """
    Public tokenizer class for the npltk toolkit.

    Parameters
    ----------
    split_into_sentences : bool
        If True (default), ``tokenize_sentences`` groups tokens by sentence.
    keep_punct : bool
        If True (default), punctuation is emitted as PUNCT tokens.
    """

    def __init__(self, *, split_into_sentences: bool = True, keep_punct: bool = True):
        self.split_into_sentences = split_into_sentences
        self.keep_punct = keep_punct

    def tokenize(self, text: str) -> List[Token]:
        """Tokenize the full text (no sentence grouping)."""
        return tokenize_words(text, keep_punct=self.keep_punct)

    def tokenize_sentences(self, text: str) -> List[TokenizedSentence]:
        """Split sentences then tokenize each sentence, returning global spans."""
        if not self.split_into_sentences:
            toks = tokenize_words(text, keep_punct=self.keep_punct)
            return [TokenizedSentence(sentence=text, start=0, end=len(text), tokens=toks)]

        spans = split_sentences(text)
        out: List[TokenizedSentence] = []

        for s in spans:
            local_tokens = tokenize_words(s.text, keep_punct=self.keep_punct)

            # shift token spans from sentence-local to global offsets
            global_tokens = [
                Token(t.text, t.start + s.start, t.end + s.start, t.type) for t in local_tokens
            ]

            out.append(TokenizedSentence(sentence=s.text, start=s.start, end=s.end, tokens=global_tokens))

        return out

    def detokenize(self, tokens: List[Token]) -> str:
        """Reconstruct text from tokens."""
        return detokenize_tokens(tokens)
