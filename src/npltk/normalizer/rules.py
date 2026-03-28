"""
rules.py
--------
All normalization rules for Nepali text.

Each rule is a dataclass with a ``name`` attribute and an ``apply(text)``
method that returns ``(transformed_text, metadata_dict)``.

Rules are designed to be composed in a pipeline (see ``core.Normalizer``).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Set


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

DEVANAGARI_WORD = re.compile(r"^[\u0900-\u097F]+$")

SPACE_MAP = {
    "\u00A0": " ",   # NBSP
    "\u202F": " ",   # narrow NBSP
    "\u2009": " ",   # thin space
    "\u200B": "",    # ZWSP → remove
    "\uFEFF": "",    # BOM → remove
}

# ── Nepali digit ↔ ASCII digit ──────────────────────────────────────────────
NEPALI_TO_ASCII = str.maketrans("०१२३४५६७८९", "0123456789")
ASCII_TO_NEPALI = str.maketrans("0123456789", "०१२३४५६७८९")

# ── Quotation marks → standard ASCII quotes ─────────────────────────────────
QUOTE_MAP = {
    "\u201C": '"',   # left double quotation mark  "
    "\u201D": '"',   # right double quotation mark "
    "\u2018": "'",   # left single quotation mark  '
    "\u2019": "'",   # right single quotation mark '
    "\u00AB": '"',   # left guillemet  «
    "\u00BB": '"',   # right guillemet »
    "\u201E": '"',   # double low-9 quotation mark „
    "\u201A": "'",   # single low-9 quotation mark ‚
}

# ── Common Nepali abbreviations that contain a dot ───────────────────────────
# These should NOT trigger sentence splits.
ABBREVIATIONS: Set[str] = {
    "डा.", "प्रा.", "श्री.", "सु.", "श्रीमती.", "प्रो.",
    "डि.", "नं.", "क.", "ख.", "ग.", "घ.",
    "ई.", "पू.", "उ.", "द.", "रु.",
    "मा.वि.", "ला.वि.", "प्र.अ.", "उ.म.", "जि.",
}

# ── Expanded postposition list  (sorted longest-first for greedy match) ──────
POSTPOSITIONS = sorted({
    # locative / ablative / dative / instrumental
    "देखि", "सम्म", "बाट", "लाई", "सँग", "द्वारा",
    "प्रति", "अनुसार", "भन्दा", "बारेमा", "बारे",
    # genitive
    "को", "का", "की",
    # locative
    "मा",
    # ergative / emphatic / pluralizer
    "ले", "त", "नि", "नै", "पनि",
    "हरू", "हरु",               # plural suffix (both spellings)
    # spatial postpositions
    "भित्र", "बाहिर", "माथि", "तल",
    "अगाडि", "पछाडि", "नजिक", "बिच", "बीच",
    # others
    "सित", "तिर", "भरि", "भर",
    "विना", "बिना",
    "लागि",
    "मुनि",
    "वरिपरि",
}, key=len, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 1 — Unicode NFC
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UnicodeNFC:
    """Normalize text to NFC form (canonical decomposition → composition)."""
    name: str = "unicode_nfc"

    def apply(self, text: str) -> Tuple[str, Dict[str, Any]]:
        out = unicodedata.normalize("NFC", text)
        return out, {"changed": out != text}


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 2 — Whitespace normalization
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WhitespaceNormalize:
    """Collapse exotic/invisible spaces & multiple blanks into single space."""
    name: str = "whitespace_normalize"

    def apply(self, text: str) -> Tuple[str, Dict[str, Any]]:
        before = text
        for k, v in SPACE_MAP.items():
            text = text.replace(k, v)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        return text, {"changed": before != text}


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 3 — Remove invisible control characters
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RemoveInvisibleChars:
    """Strip C0 control chars (except \\n and \\t)."""
    name: str = "remove_invisible_chars"

    def apply(self, text: str) -> Tuple[str, Dict[str, Any]]:
        before = text
        removed = 0
        out_chars = []
        for ch in text:
            cat = unicodedata.category(ch)
            if cat == "Cc" and ch not in ("\n", "\t"):
                removed += 1
                continue
            out_chars.append(ch)
        out = "".join(out_chars)
        return out, {"removed": removed, "changed": before != out}


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 4 — ZWJ / ZWNJ cleanup
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ZWJZWNJCleanup:
    """Remove zero-width joiner / non-joiner characters."""
    name: str = "zwj_zwnj_cleanup"

    def apply(self, text: str) -> Tuple[str, Dict[str, Any]]:
        before = text
        out = text.replace("\u200D", "").replace("\u200C", "")
        return out, {"changed": before != out}


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 5 — Halant (virama) cleanup
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HalantCleanup:
    """Fix halant followed by whitespace or repeated halants."""
    name: str = "halant_cleanup"

    def apply(self, text: str) -> Tuple[str, Dict[str, Any]]:
        before = text
        text = re.sub(r"्\s+", "्", text)    # halant + extraneous spaces
        text = re.sub(r"्{2,}", "्", text)   # repeated halant
        return text, {"changed": before != text}


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 6 — Diacritic deduplication
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DiacriticDedupe:
    """Remove repeated anusvara (ं) and chandrabindu (ँ)."""
    name: str = "diacritic_dedupe"

    def apply(self, text: str) -> Tuple[str, Dict[str, Any]]:
        before = text
        text = re.sub(r"ं{2,}", "ं", text)
        text = re.sub(r"ँ{2,}", "ँ", text)
        return text, {"changed": before != text}


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 7 — Quotation mark normalization  [NEW]
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuotationNormalize:
    """Normalize fancy/curly quotation marks to plain ASCII quotes."""
    name: str = "quotation_normalize"

    def apply(self, text: str) -> Tuple[str, Dict[str, Any]]:
        before = text
        for fancy, plain in QUOTE_MAP.items():
            text = text.replace(fancy, plain)
        return text, {"changed": before != text}


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 8 — Nepali digit normalization  [NEW]
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NepaliDigitNormalize:
    """
    Unify digit representations to avoid duplicate vocab entries.
    Default direction: Nepali → ASCII  (२०२६ → 2026).
    Set ``to_nepali=True`` for the reverse.
    """
    name: str = "digit_normalize"
    to_nepali: bool = False

    def apply(self, text: str) -> Tuple[str, Dict[str, Any]]:
        before = text
        table = ASCII_TO_NEPALI if self.to_nepali else NEPALI_TO_ASCII
        text = text.translate(table)
        return text, {"changed": before != text}


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 9 — Repetition compression (social media)  [NEW]
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RepetitionCompress:
    """
    Compress characters repeated more than ``max_repeat`` times.
    Handles both single-char runs (हाहाहाहाहा) and multi-char
    syllable-level repetitions.

    Examples:
        हाहाहाहाहा → हाहा
        !!!!!!!    → !!
        aaaaaaa    → aa
    """
    name: str = "repetition_compress"
    max_repeat: int = 2

    def apply(self, text: str) -> Tuple[str, Dict[str, Any]]:
        before = text
        n = self.max_repeat
        # Single-char runs: (x){n+1,} → x * n
        text = re.sub(r"(.)\1{" + str(n) + r",}", r"\1" * n, text)
        # Two-char syllable runs: (xy){n+1,} → xy * n   (catches हाहाहा…)
        text = re.sub(r"(.{2})\1{" + str(n) + r",}", r"\1" * n, text)
        return text, {"changed": before != text}


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 10 — Script-boundary splitter  [NEW]
# ═══════════════════════════════════════════════════════════════════════════════

# Patterns to detect Devanagari and Latin blocks
_DEV_CHAR = re.compile(r"[\u0900-\u097F]")
_LAT_CHAR = re.compile(r"[A-Za-z]")


@dataclass
class ScriptBoundarySplit:
    """
    Insert a space at script transitions within a single 'word'.
    Handles mixed tokens common in Nepali social media:
        Facebookमा → Facebook मा
        lockdownको → lockdown को
        #NepaliPride → # NepaliPride (handled separately by HashtagSplit)
    """
    name: str = "script_boundary_split"

    def apply(self, text: str) -> Tuple[str, Dict[str, Any]]:
        before = text
        splits = 0

        words = text.split(" ")
        out: List[str] = []

        for word in words:
            if not word:
                out.append(word)
                continue

            # Check if the word contains BOTH Devanagari and Latin chars
            has_dev = bool(_DEV_CHAR.search(word))
            has_lat = bool(_LAT_CHAR.search(word))

            if has_dev and has_lat:
                # Insert space at every Dev↔Lat boundary
                new_word = self._split_at_boundaries(word)
                out.append(new_word)
                if new_word != word:
                    splits += 1
            else:
                out.append(word)

        after = " ".join(out)
        return after, {"splits": splits, "changed": before != after}

    @staticmethod
    def _split_at_boundaries(word: str) -> str:
        """Insert spaces at Devanagari ↔ Latin transitions."""
        if len(word) < 2:
            return word

        parts: List[str] = [word[0]]
        for i in range(1, len(word)):
            prev_dev = bool(_DEV_CHAR.match(word[i - 1]))
            curr_dev = bool(_DEV_CHAR.match(word[i]))
            prev_lat = bool(_LAT_CHAR.match(word[i - 1]))
            curr_lat = bool(_LAT_CHAR.match(word[i]))

            # Transition: Devanagari → Latin  or  Latin → Devanagari
            if (prev_dev and curr_lat) or (prev_lat and curr_dev):
                parts.append(" ")
            parts.append(word[i])

        return "".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 11 — Hashtag / Mention splitter  [NEW]
# ═══════════════════════════════════════════════════════════════════════════════

_HASHTAG_RE = re.compile(r"([#@])([\w\u0900-\u097F]+)")


@dataclass
class HashtagMentionSplit:
    """
    Separate # and @ symbols from the word that follows them.
        #नेपाल → # नेपाल
        @username → @ username
    """
    name: str = "hashtag_mention_split"

    def apply(self, text: str) -> Tuple[str, Dict[str, Any]]:
        before = text
        text = _HASHTAG_RE.sub(r"\1 \2", text)
        count = len(_HASHTAG_RE.findall(before))
        return text, {"splits": count, "changed": before != text}


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 12 — Abbreviation protector  [NEW]
# ═══════════════════════════════════════════════════════════════════════════════

# Build a regex that replaces known abbreviation dots with a placeholder
# so that downstream sentence splitters don't break on them.
_ABBR_PLACEHOLDER = "\uFFF9"   # private-use char, extremely unlikely in text


@dataclass
class AbbreviationProtect:
    """
    Replace dots inside known abbreviations with a placeholder so that
    sentence splitters don't break on them.

    The companion ``AbbreviationRestore`` rule should run AFTER sentence
    splitting to put the dots back.

    Example:
        डा. शर्मा → डा\uFFF9 शर्मा  (internally)
    """
    name: str = "abbreviation_protect"

    def apply(self, text: str) -> Tuple[str, Dict[str, Any]]:
        before = text
        protected = 0
        for abbr in sorted(ABBREVIATIONS, key=len, reverse=True):
            if abbr in text:
                safe = abbr.replace(".", _ABBR_PLACEHOLDER)
                text = text.replace(abbr, safe)
                protected += 1
        return text, {"protected": protected, "changed": before != text}


@dataclass
class AbbreviationRestore:
    """Restore abbreviation dots that were protected earlier."""
    name: str = "abbreviation_restore"

    def apply(self, text: str) -> Tuple[str, Dict[str, Any]]:
        before = text
        text = text.replace(_ABBR_PLACEHOLDER, ".")
        return text, {"changed": before != text}


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 13 — Postposition split  (IMPROVED — expanded list)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PostpositionSplit:
    """
    Split agglutinated Nepali postpositions from their root words.
        नेपालबाट → नेपाल बाट
        घरहरूमा  → घरहरू मा
    """
    name: str = "postposition_split"
    min_root_len: int = 2

    def apply(self, text: str) -> Tuple[str, Dict[str, Any]]:
        before = text
        tokens = text.split(" ")
        out: List[str] = []
        splits = 0

        for tok in tokens:
            if not tok or not DEVANAGARI_WORD.match(tok) or len(tok) < 3:
                out.append(tok)
                continue

            did_split = False
            for suf in POSTPOSITIONS:
                if tok.endswith(suf) and len(tok) > len(suf) + self.min_root_len - 1:
                    root = tok[: -len(suf)]
                    if len(root) >= self.min_root_len:
                        out.extend([root, suf])
                        splits += 1
                        did_split = True
                        break

            if not did_split:
                out.append(tok)

        after = " ".join(out)
        return after, {"splits": splits, "changed": before != after}