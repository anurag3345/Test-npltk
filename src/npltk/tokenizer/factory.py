from __future__ import annotations

from typing import Callable, Literal, Optional, Union

from .hybrid_tokenizer import NepaliHybridTokenizer
from .tokenizer import NepaliTokenizer

TokenizerMode = Literal["hybrid", "rule"]
TokenizerLike = Union[NepaliTokenizer, NepaliHybridTokenizer]


def create_tokenizer(
    *,
    mode: TokenizerMode = "hybrid",
    split_into_sentences: bool = True,
    keep_punct: bool = True,
    model_path: Optional[str] = None,
    subword: bool = True,
    preprocess: Optional[Callable[[str], str]] = None,
    fallback_to_rule: bool = True,
) -> TokenizerLike:
    """
    Create a tokenizer using one unified API.

    Parameters
    ----------
    mode : {"hybrid", "rule"}
        ``hybrid`` (default) uses SentencePiece-backed tokenization.
        ``rule`` uses pure rule-based tokenization.
    split_into_sentences : bool
        Enable sentence grouping for ``tokenize_sentences``.
    keep_punct : bool
        Preserve punctuation tokens.
    model_path : str, optional
        Path to SentencePiece model when using hybrid mode.
    subword : bool
        Enable subword segmentation in hybrid mode.
    preprocess : callable, optional
        Optional preprocessor used before hybrid tokenization.
    fallback_to_rule : bool
        If True, hybrid init failures fall back to NepaliTokenizer.
    """
    if mode == "rule":
        return NepaliTokenizer(
            split_into_sentences=split_into_sentences,
            keep_punct=keep_punct,
        )

    if mode != "hybrid":
        raise ValueError("mode must be either 'hybrid' or 'rule'")

    try:
        return NepaliHybridTokenizer(
            model_path=model_path,
            split_into_sentences=split_into_sentences,
            keep_punct=keep_punct,
            subword=subword,
            preprocess=preprocess,
        )
    except Exception:
        if not fallback_to_rule:
            raise
        return NepaliTokenizer(
            split_into_sentences=split_into_sentences,
            keep_punct=keep_punct,
        )
