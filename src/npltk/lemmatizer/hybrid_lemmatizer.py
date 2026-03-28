from __future__ import annotations

from .lemmatizer import Lemmatizer


class HybridLemmatizer(Lemmatizer):
    """Backward-compatible alias for the lemmatizer pipeline class."""


__all__ = ["HybridLemmatizer"]
