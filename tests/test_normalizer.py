"""
test_normalizer.py
------------------
Tests for every normalization rule in npltk.normalizer.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from npltk.normalizer import build_normalizer, NormalizerConfig
from npltk.normalizer.rules import (
    UnicodeNFC,
    WhitespaceNormalize,
    RemoveInvisibleChars,
    ZWJZWNJCleanup,
    HalantCleanup,
    DiacriticDedupe,
    QuotationNormalize,
    NepaliDigitNormalize,
    RepetitionCompress,
    ScriptBoundarySplit,
    HashtagMentionSplit,
    AbbreviationProtect,
    AbbreviationRestore,
    PostpositionSplit,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 1: Unicode NFC
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnicodeNFC:
    def test_unchanged(self):
        rule = UnicodeNFC()
        out, meta = rule.apply("नेपाल")
        assert out == "नेपाल"

    def test_nfc_normalizes(self):
        # NFD form of 'ü' (u + combining diaeresis)
        nfd = "u\u0308"
        rule = UnicodeNFC()
        out, _ = rule.apply(nfd)
        assert out == "ü"


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 2: Whitespace normalization
# ═══════════════════════════════════════════════════════════════════════════════

class TestWhitespace:
    def test_multiple_spaces(self):
        rule = WhitespaceNormalize()
        out, meta = rule.apply("hello    world")
        assert out == "hello world"
        assert meta["changed"] is True

    def test_nbsp_removal(self):
        rule = WhitespaceNormalize()
        out, _ = rule.apply("hello\u00A0world")
        assert out == "hello world"

    def test_zwsp_removal(self):
        rule = WhitespaceNormalize()
        out, _ = rule.apply("hello\u200Bworld")
        assert out == "helloworld"

    def test_multiple_newlines(self):
        rule = WhitespaceNormalize()
        out, _ = rule.apply("a\n\n\n\nb")
        assert out == "a\n\nb"


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 3: Invisible chars
# ═══════════════════════════════════════════════════════════════════════════════

class TestInvisibleChars:
    def test_removes_control_chars(self):
        rule = RemoveInvisibleChars()
        out, meta = rule.apply("hello\x00world")
        assert out == "helloworld"
        assert meta["removed"] == 1

    def test_keeps_newline_tab(self):
        rule = RemoveInvisibleChars()
        out, _ = rule.apply("hello\n\tworld")
        assert out == "hello\n\tworld"


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 4: ZWJ / ZWNJ
# ═══════════════════════════════════════════════════════════════════════════════

class TestZWJ:
    def test_removes_zwj(self):
        rule = ZWJZWNJCleanup()
        out, _ = rule.apply("नेपाल\u200Dबाट")
        assert out == "नेपालबाट"

    def test_removes_zwnj(self):
        rule = ZWJZWNJCleanup()
        out, _ = rule.apply("test\u200Ctext")
        assert out == "testtext"


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 5: Halant cleanup
# ═══════════════════════════════════════════════════════════════════════════════

class TestHalant:
    def test_halant_space(self):
        rule = HalantCleanup()
        out, _ = rule.apply("विद्  यार्थी")
        assert "् " not in out

    def test_repeated_halant(self):
        rule = HalantCleanup()
        out, _ = rule.apply("विद््यार्थी")
        assert "््" not in out


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 6: Diacritic dedupe
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiacriticDedupe:
    def test_anusvara(self):
        rule = DiacriticDedupe()
        out, _ = rule.apply("संंस्कृति")
        assert out == "संस्कृति"

    def test_chandrabindu(self):
        rule = DiacriticDedupe()
        out, _ = rule.apply("गएँँ")
        assert out == "गएँ"


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 7: Quotation normalization
# ═══════════════════════════════════════════════════════════════════════════════

class TestQuotation:
    def test_curly_double(self):
        rule = QuotationNormalize()
        out, _ = rule.apply("\u201Cहेल्लो\u201D")
        assert out == '"हेल्लो"'

    def test_curly_single(self):
        rule = QuotationNormalize()
        out, _ = rule.apply("\u2018yes\u2019")
        assert out == "'yes'"

    def test_guillemets(self):
        rule = QuotationNormalize()
        out, _ = rule.apply("«test»")
        assert out == '"test"'


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 8: Nepali digit normalization
# ═══════════════════════════════════════════════════════════════════════════════

class TestDigitNormalize:
    def test_nepali_to_ascii(self):
        rule = NepaliDigitNormalize(to_nepali=False)
        out, _ = rule.apply("मिति २०२६/०१/३१")
        assert out == "मिति 2026/01/31"

    def test_ascii_to_nepali(self):
        rule = NepaliDigitNormalize(to_nepali=True)
        out, _ = rule.apply("Year 2026")
        assert out == "Year २०२६"

    def test_no_change_needed(self):
        rule = NepaliDigitNormalize(to_nepali=False)
        out, meta = rule.apply("hello world")
        assert out == "hello world"
        assert meta["changed"] is False


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 9: Repetition compression
# ═══════════════════════════════════════════════════════════════════════════════

class TestRepetition:
    def test_repeated_chars(self):
        rule = RepetitionCompress()
        out, _ = rule.apply("!!!!!!!!")
        assert out == "!!"

    def test_repeated_vowel(self):
        rule = RepetitionCompress()
        out, _ = rule.apply("राम्रोोोो")
        assert out == "राम्रोो"

    def test_normal_text_unchanged(self):
        rule = RepetitionCompress()
        out, meta = rule.apply("नेपाल")
        assert out == "नेपाल"


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 10: Script-boundary split
# ═══════════════════════════════════════════════════════════════════════════════

class TestScriptBoundary:
    def test_latin_devanagari(self):
        rule = ScriptBoundarySplit()
        out, meta = rule.apply("Facebookमा")
        assert "Facebook" in out
        assert "मा" in out
        assert meta["splits"] == 1

    def test_devanagari_latin(self):
        rule = ScriptBoundarySplit()
        out, _ = rule.apply("lockdownको")
        assert "lockdown" in out
        assert "को" in out

    def test_pure_devanagari_unchanged(self):
        rule = ScriptBoundarySplit()
        out, meta = rule.apply("नेपालबाट")
        assert out == "नेपालबाट"
        assert meta["splits"] == 0

    def test_pure_latin_unchanged(self):
        rule = ScriptBoundarySplit()
        out, meta = rule.apply("Facebook")
        assert out == "Facebook"


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 11: Hashtag / Mention split
# ═══════════════════════════════════════════════════════════════════════════════

class TestHashtag:
    def test_hashtag(self):
        rule = HashtagMentionSplit()
        out, _ = rule.apply("#नेपाल")
        assert out == "# नेपाल"

    def test_mention(self):
        rule = HashtagMentionSplit()
        out, _ = rule.apply("@username")
        assert out == "@ username"

    def test_no_hashtag(self):
        rule = HashtagMentionSplit()
        out, meta = rule.apply("hello world")
        assert out == "hello world"


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 12: Abbreviation protect / restore
# ═══════════════════════════════════════════════════════════════════════════════

class TestAbbreviation:
    def test_protect_and_restore(self):
        protect = AbbreviationProtect()
        restore = AbbreviationRestore()
        text = "डा. शर्मा"

        protected, _ = protect.apply(text)
        # Dot should be replaced with placeholder
        assert "." not in protected
        assert "शर्मा" in protected

        restored, _ = restore.apply(protected)
        assert restored == text

    def test_multiple_abbreviations(self):
        protect = AbbreviationProtect()
        restore = AbbreviationRestore()
        text = "डा. शर्मा र श्री. राम"

        protected, meta = protect.apply(text)
        assert meta["protected"] >= 2

        restored, _ = restore.apply(protected)
        assert restored == text


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule 13: Postposition split
# ═══════════════════════════════════════════════════════════════════════════════

class TestPostposition:
    def test_baat(self):
        rule = PostpositionSplit()
        out, meta = rule.apply("नेपालबाट")
        assert "नेपाल" in out and "बाट" in out
        assert meta["splits"] == 1

    def test_ma(self):
        rule = PostpositionSplit()
        out, _ = rule.apply("काठमाडौंमा")
        assert "काठमाडौं" in out and "मा" in out

    def test_haru_ma(self):
        rule = PostpositionSplit()
        out, _ = rule.apply("घरहरूमा")
        # Should split at least one postposition
        assert "मा" in out or "हरू" in out

    def test_short_word_no_split(self):
        rule = PostpositionSplit()
        out, meta = rule.apply("मा")
        assert out == "मा"
        assert meta["splits"] == 0

    def test_expanded_postpositions(self):
        """Test newly added postpositions."""
        rule = PostpositionSplit()
        # Test 'द्वारा'
        out, meta = rule.apply("सरकारद्वारा")
        assert "सरकार" in out and "द्वारा" in out

    def test_bhanda(self):
        rule = PostpositionSplit()
        out, _ = rule.apply("यसभन्दा")
        assert "भन्दा" in out


# ═══════════════════════════════════════════════════════════════════════════════
#  Integration: build_normalizer full pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    def test_default_config(self):
        norm = build_normalizer()
        result = norm.normalize("घरमा\u200B  नेपालबाट\u200D गएँ")
        assert "\u200B" not in result.text
        assert "\u200D" not in result.text

    def test_mixed_script(self):
        norm = build_normalizer()
        result = norm.normalize("Facebookमा photo हाल्नु")
        assert "Facebook" in result.text
        assert "मा" in result.text

    def test_social_media(self):
        norm = build_normalizer()
        result = norm.normalize("#NepaliPride हाहाहाहाहा!!!!!!")
        text = result.text
        assert "# " in text                  # hashtag split
        assert text.count("!") <= 2           # repetition compressed
        assert text.count("ह") < 10           # syllable compressed

    def test_abbreviation_survives_pipeline(self):
        norm = build_normalizer()
        result = norm.normalize("डा. शर्मा")
        assert "डा." in result.text

    def test_digit_normalization(self):
        norm = build_normalizer()
        result = norm.normalize("२०२६ सालमा")
        assert "2026" in result.text

    def test_empty_string(self):
        norm = build_normalizer()
        result = norm.normalize("")
        assert result.text == ""

    def test_config_disable_digits(self):
        cfg = NormalizerConfig(digit_normalize=False)
        norm = build_normalizer(cfg)
        result = norm.normalize("मिति २०२६")
        assert "२०२६" in result.text    # digits NOT converted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
