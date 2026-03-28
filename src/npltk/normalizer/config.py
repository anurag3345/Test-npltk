"""
config.py
---------
Toggle individual normalization rules on/off via ``NormalizerConfig``.
"""
from dataclasses import dataclass


@dataclass
class NormalizerConfig:
    """All flags default to True — disable individual rules as needed."""

    # ── Original rules ───────────────────────────────────────────────────
    unicode_nfc: bool = True
    whitespace: bool = True
    invisible_chars: bool = True
    zwj_zwnj: bool = True
    halant_cleanup: bool = True
    diacritic_dedupe: bool = True

    # ── New rules ────────────────────────────────────────────────────────
    quotation_normalize: bool = True
    digit_normalize: bool = True
    digit_to_nepali: bool = False     # direction: False=Nepali→ASCII, True=ASCII→Nepali
    repetition_compress: bool = True
    script_boundary_split: bool = True
    hashtag_mention_split: bool = True
    abbreviation_protect: bool = True

    # ── Postposition split (unchanged) ───────────────────────────────────
    postposition_split: bool = True