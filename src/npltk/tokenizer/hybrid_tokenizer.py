"""
hybrid_tokenizer.py
-------------------
Stage-2 router: combines the rule engine pre-tokens with a trained
SentencePiece Unigram model to produce sub-word-aware tokens for
Devanagari text while leaving all other token types untouched.

Optionally integrates the StopWordRemover to filter Nepali stopwords
from the tokenization output.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from .rule_engine import PreToken, pre_tokenize
from .sentence_splitter import split_sentences, SentenceSpan
from .types import Token, TokenType, TokenizedSentence


# ---------------------------------------------------------------------------
# Lazy import of sentencepiece so the package stays importable even if
# sentencepiece is not installed (users get a clear error only when they
# try to use the hybrid tokenizer, not at import time).
# ---------------------------------------------------------------------------
def _load_sp(model_path: str):
    try:
        import sentencepiece as spm  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "sentencepiece is required for NepaliHybridTokenizer. "
            "Install it with:  pip install sentencepiece"
        ) from exc
    proc = spm.SentencePieceProcessor()
    proc.load(model_path)
    return proc


# Default model path — bundled inside the package
_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "models", "nepali_tokenizer.model")


# ---------------------------------------------------------------------------
# Core helper: expand a WORD_DEV PreToken using the SP model
# ---------------------------------------------------------------------------
def _expand_dev_token(pt: PreToken, sp) -> List[Token]:
    """
    Run SentencePiece on a single Devanagari word chunk.
    Reconstructs per-piece character offsets from the original span.
    """
    pieces: List[str] = sp.encode(pt.text, out_type=str)

    # If SP returns the word as-is (single piece), emit as WORD_DEV
    if len(pieces) == 1:
        clean = pieces[0].lstrip("\u2581")
        return [Token(clean or pt.text, pt.start, pt.end, TokenType.WORD_DEV)]

    tokens: List[Token] = []
    cursor = pt.start            # current position in original string
    raw = pt.text

    for i, piece in enumerate(pieces):
        clean = piece.lstrip("\u2581")   # remove sentencepiece space-prefix marker
        if not clean:
            continue

        # Find piece in the remaining raw text (left-to-right, greedy)
        idx = raw.find(clean)
        if idx == -1:
            # fallback: emit with best-guess span
            tokens.append(Token(clean, cursor, cursor + len(clean), TokenType.SUBWORD_DEV))
            cursor += len(clean)
        else:
            abs_start = pt.start + idx
            abs_end   = abs_start + len(clean)
            tokens.append(Token(clean, abs_start, abs_end, TokenType.SUBWORD_DEV))
            # Advance past this piece in the local raw string
            raw    = raw[idx + len(clean):]
            cursor = abs_end

    return tokens if tokens else [Token(pt.text, pt.start, pt.end, TokenType.WORD_DEV)]


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------
class NepaliHybridTokenizer:
    """
    Hybrid rule-based + SentencePiece tokenizer for Nepali text.

    Stage 1 — rule engine:
        Fast regex pre-tokenizer classifies chunks by script type.

    Stage 2 — model router:
        Devanagari (WORD_DEV) chunks → SentencePiece Unigram sub-word model.
        Latin / numbers / punctuation / emoji → passed through unchanged.

    Stopword removal is available via the ``filter_stopwords()`` method,
    which can be called on any token list after tokenization.

    Parameters
    ----------
    model_path : str, optional
        Path to the trained ``nepali_sp.model`` file.
        Defaults to the model bundled inside the package.
    split_into_sentences : bool
        If True (default), ``tokenize_sentences`` groups tokens by sentence.
    keep_punct : bool
        If True (default), punctuation is emitted as PUNCT tokens.
    subword : bool
        If True (default), Devanagari words are segmented into sub-word
        pieces by the SP model.  Set to False to fall back to pure rule-based.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        split_into_sentences: bool = True,
        keep_punct: bool = True,
        subword: bool = True,
    ) -> None:
        self.split_into_sentences = split_into_sentences
        self.keep_punct = keep_punct
        self.subword = subword
        self._sp = None

        if subword:
            path = model_path or _DEFAULT_MODEL
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"SentencePiece model not found at: {path}\n"
                    "Train a model first with:  python scripts/train_tokenizer.py\n"
                    "Then place 'nepali_sp.model' in src/npltk/tokenizer/models/"
                )
            self._sp = _load_sp(path)

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------
    def _pre_to_tokens(self, pre_tokens: List[PreToken]) -> List[Token]:
        """Convert a flat list of PreTokens to final Tokens."""
        out: List[Token] = []
        for pt in pre_tokens:
            if pt.type == TokenType.WORD_DEV and self.subword and self._sp is not None:
                out.extend(_expand_dev_token(pt, self._sp))
            else:
                out.append(Token(pt.text, pt.start, pt.end, pt.type))
        return out

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------
    def tokenize(self, text: str) -> List[Token]:
        """Tokenize *text* into a flat list of Tokens (no sentence grouping)."""
        pre = pre_tokenize(text, keep_punct=self.keep_punct)
        return self._pre_to_tokens(pre)

    def tokenize_sentences(self, text: str) -> List[TokenizedSentence]:
        """
        Split *text* into sentences then tokenize each one.
        Token spans are global (relative to the original *text*).
        """
        if not self.split_into_sentences:
            toks = self.tokenize(text)
            return [TokenizedSentence(sentence=text, start=0, end=len(text), tokens=toks)]

        spans: List[SentenceSpan] = split_sentences(text)
        result: List[TokenizedSentence] = []

        for s in spans:
            pre = pre_tokenize(s.text, keep_punct=self.keep_punct)
            local_tokens = self._pre_to_tokens(pre)
            # shift local offsets → global offsets
            global_tokens = [
                Token(t.text, t.start + s.start, t.end + s.start, t.type)
                for t in local_tokens
            ]
            result.append(
                TokenizedSentence(sentence=s.text, start=s.start, end=s.end, tokens=global_tokens)
            )

        return result

    def detokenize(self, tokens: List[Token]) -> str:
        """
        Reconstruct text from tokens.
        Sub-word pieces are joined directly (no space before SUBWORD_DEV
        unless it is the first token or follows punctuation).
        """
        parts: List[str] = []
        for i, tok in enumerate(tokens):
            if i == 0:
                parts.append(tok.text)
            elif tok.type == TokenType.SUBWORD_DEV:
                parts.append(tok.text)          # join sub-words without space
            elif tok.type == TokenType.PUNCT:
                parts.append(tok.text)          # no space before punctuation
            elif tokens[i - 1].type == TokenType.PUNCT and tokens[i-1].text in ("(", "[", "{", "\u00ab", "\u201c", "\u2018"):
                parts.append(tok.text)          # no space after opening bracket
            else:
                parts.append(" " + tok.text)
        return "".join(parts)
