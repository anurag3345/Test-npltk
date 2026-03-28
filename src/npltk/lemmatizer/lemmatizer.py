from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional

from .dict_lookup import LemmaDictionary
from .rule_stripper import RuleStripper


class Lemmatizer:
    def __init__(
        self,
        dictionary_path: str | Path | None = None,
        *,
        cache_size: int = 4096,
        min_root_len: int = 2,
    ) -> None:
        if dictionary_path is None:
            dictionary_path = Path(__file__).parent / "data" / "lemma_dict.json"

        self.dictionary = LemmaDictionary(dictionary_path)
        self.rule_stripper = RuleStripper(min_root_len=min_root_len)
        self.cache_size = max(1, cache_size)
        self.cache: "OrderedDict[str, str]" = OrderedDict()

    def lemmatize(self, word: str) -> str:
        cached = self._cache_get(word)
        if cached is not None:
            return cached

        lemma = self.dictionary.lookup(word)
        if lemma is not None:
            self._cache_set(word, lemma)
            return lemma

        lemma = self.rule_stripper.lemmatize(word)
        if lemma is not None:
            self._cache_set(word, lemma)
            return lemma

        self._cache_set(word, word)
        return word

    def lemmatize_many(self, words: list[str]) -> list[str]:
        return [self.lemmatize(w) for w in words]

    def _cache_get(self, word: str) -> Optional[str]:
        lemma = self.cache.get(word)
        if lemma is None:
            return None
        self.cache.move_to_end(word)
        return lemma

    def _cache_set(self, word: str, lemma: str) -> None:
        self.cache[word] = lemma
        self.cache.move_to_end(word)
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
