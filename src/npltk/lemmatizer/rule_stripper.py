from __future__ import annotations

from typing import Optional


class RuleStripper:
    def __init__(self, min_root_len: int = 2) -> None:
        self.min_root_len = min_root_len

        noun_suffixes = [
            "हरूमा",
            "हरुमा",
            "हरूबाट",
            "हरुबाट",
            "हरूको",
            "हरुको",
            "हरूले",
            "हरुले",
            "हरू",
            "हरु",
            "बाट",
            "सँग",
            "सम्म",
            "देखि",
            "मा",
            "को",
            "का",
            "की",
            "ले",
            "लाई",
        ]
        self.noun_suffixes = sorted(noun_suffixes, key=len, reverse=True)

        verb_suffixes = [
            "दैछन्",
            "दैछ",
            "दैन",
            "दैं",
            "दै",
            "न्छन्",
            "न्छ",
            "यो",
            "े",
            "छ",
        ]
        self.verb_suffixes = sorted(verb_suffixes, key=len, reverse=True)

    def lemmatize(self, word: str) -> Optional[str]:
        noun_candidate = self._strip_noun_suffixes(word)
        if noun_candidate is not None and noun_candidate != word:
            return noun_candidate

        verb_candidate = self._strip_verb_suffixes(word)
        if verb_candidate is not None and verb_candidate != word:
            return verb_candidate

        return None

    def _strip_noun_suffixes(self, word: str) -> Optional[str]:
        current = word
        removed_any = False

        while True:
            matched = False
            for suf in self.noun_suffixes:
                if current.endswith(suf):
                    candidate = current[: -len(suf)]
                    if len(candidate) < self.min_root_len:
                        continue
                    current = candidate
                    removed_any = True
                    matched = True
                    break
            if not matched:
                break

        if removed_any and len(current) >= self.min_root_len:
            return current
        return None

    def _strip_verb_suffixes(self, word: str) -> Optional[str]:
        working = word
        removed_any = False

        while True:
            matched = False
            for suf in self.verb_suffixes:
                if working.endswith(suf):
                    candidate = working[: -len(suf)]
                    if len(candidate) < self.min_root_len:
                        continue
                    working = candidate
                    removed_any = True
                    matched = True
                    break
            if not matched:
                break

        if not removed_any:
            return None

        normalized = self._normalize_verb_stem(working)
        if normalized is None or len(normalized) < self.min_root_len:
            return None
        return normalized

    def _normalize_verb_stem(self, stem: str) -> Optional[str]:
        if len(stem) < self.min_root_len:
            return None

        if stem == "गर":
            return "गर्नु"

        if stem.endswith("ा"):
            return stem + "नु"

        if stem.endswith("र"):
            return stem + "्नु"

        if stem.endswith("न"):
            return stem + "ु"

        return stem + "नु"
