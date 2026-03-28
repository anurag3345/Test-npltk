"""
npltk.normalizer
----------------
Public API for the Nepali text normalizer.

Usage::

    from npltk.normalizer import build_normalizer

    normalizer = build_normalizer()
    result = normalizer.normalize("घरमा\\u200B  नेपालबाट गएँ")
    print(result.text)          # normalized string
    print(result.transforms)    # list of applied transformations
"""
from .core import Normalizer, NormResult
from .config import NormalizerConfig
from .rules import (
    # Original rules
    UnicodeNFC,
    WhitespaceNormalize,
    RemoveInvisibleChars,
    ZWJZWNJCleanup,
    HalantCleanup,
    DiacriticDedupe,
    PostpositionSplit,
    # New rules
    QuotationNormalize,
    NepaliDigitNormalize,
    RepetitionCompress,
    ScriptBoundarySplit,
    HashtagMentionSplit,
    AbbreviationProtect,
    AbbreviationRestore,
)


def build_normalizer(cfg: NormalizerConfig | None = None) -> Normalizer:
    """
    Build a normalizer with rules ordered for correctness.

    Rule ordering matters:
    1. Unicode NFC first (canonical form for all subsequent regex)
    2. Whitespace / invisible / ZWJ cleanup (remove noise)
    3. Halant + diacritic fixes (Devanagari-specific corrections)
    4. Quotation + digit normalization (standardise representations)
    5. Repetition compression (social media de-noise)
    6. Abbreviation protection (before any splitting)
    7. Script-boundary + hashtag splitting (mixed-script handling)
    8. Postposition split last (requires clean words)
    """
    cfg = cfg or NormalizerConfig()
    rules = []

    # ── Phase 1: Unicode & whitespace ────────────────────────────────────
    if cfg.unicode_nfc:
        rules.append(UnicodeNFC())
    if cfg.whitespace:
        rules.append(WhitespaceNormalize())
    if cfg.invisible_chars:
        rules.append(RemoveInvisibleChars())
    if cfg.zwj_zwnj:
        rules.append(ZWJZWNJCleanup())

    # ── Phase 2: Devanagari-specific fixes ───────────────────────────────
    if cfg.halant_cleanup:
        rules.append(HalantCleanup())
    if cfg.diacritic_dedupe:
        rules.append(DiacriticDedupe())

    # ── Phase 3: Standardisation ─────────────────────────────────────────
    if cfg.quotation_normalize:
        rules.append(QuotationNormalize())
    if cfg.digit_normalize:
        rules.append(NepaliDigitNormalize(to_nepali=cfg.digit_to_nepali))

    # ── Phase 4: Social media de-noise ───────────────────────────────────
    if cfg.repetition_compress:
        rules.append(RepetitionCompress())

    # ── Phase 5: Pre-split protections ───────────────────────────────────
    if cfg.abbreviation_protect:
        rules.append(AbbreviationProtect())

    # ── Phase 6: Splitting rules ─────────────────────────────────────────
    if cfg.script_boundary_split:
        rules.append(ScriptBoundarySplit())
    if cfg.hashtag_mention_split:
        rules.append(HashtagMentionSplit())
    if cfg.postposition_split:
        rules.append(PostpositionSplit())

    # ── Phase 7: Restore abbreviations ───────────────────────────────────
    if cfg.abbreviation_protect:
        rules.append(AbbreviationRestore())

    return Normalizer(rules)