import sys
import os
from typing import Any, cast
import pytest

# Add src folder to path for local run
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from npltk import create_tokenizer
from npltk.tokenizer.hybrid_tokenizer import NepaliHybridTokenizer
from npltk.tokenizer.tokenizer import NepaliTokenizer
from npltk.tokenizer.hybrid_tokenizer import _expand_dev_token
from npltk.tokenizer.rule_engine import PreToken, pre_tokenize
from npltk.tokenizer.types import Token, TokenType

def test_tokenizer_basic():
    tokenizer = NepaliTokenizer()
    text = "नेपाल एक सुन्दर देश हो।"
    tokens = tokenizer.tokenize(text)
    reconstructed = tokenizer.detokenize(tokens)

    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert reconstructed == text


def test_pre_tokenize_types_for_mixed_input():
    text = "नेपाल school २०२६ 🙂!"
    tokens = pre_tokenize(text)

    types = [t.type for t in tokens]

    assert TokenType.WORD_DEV in types
    assert TokenType.WORD_LAT in types
    assert TokenType.NUM in types
    assert TokenType.EMOJI in types
    assert TokenType.PUNCT in types


def test_expand_dev_token_preserves_monotonic_offsets_for_repeated_pieces():
    class FakeSP:
        def encode(self, text, out_type=str):
            return ["abc", "abc"]

    pt = PreToken(text="abcabc", start=10, end=16, type=TokenType.WORD_DEV)
    out = _expand_dev_token(pt, FakeSP())

    assert len(out) == 2
    assert out[0] == Token("abc", 10, 13, TokenType.SUBWORD_DEV)
    assert out[1] == Token("abc", 13, 16, TokenType.SUBWORD_DEV)


def test_shared_detokenize_behavior_for_subwords_and_punct():
    tokens = [
        Token("नेपाल", 0, 5, TokenType.WORD_DEV),
        Token("(", 5, 6, TokenType.PUNCT),
        Token("good", 6, 10, TokenType.WORD_LAT),
        Token(")", 10, 11, TokenType.PUNCT),
        Token("सु", 11, 13, TokenType.SUBWORD_DEV),
        Token("न्दर", 13, 17, TokenType.SUBWORD_DEV),
    ]

    base = NepaliTokenizer()
    hybrid = NepaliHybridTokenizer(subword=False)

    assert base.detokenize(tokens) == hybrid.detokenize(tokens)
    assert base.detokenize(tokens) == "नेपाल(good)सुन्दर"


def test_hybrid_supports_preprocess_hook():
    # The preprocess hook helps align inference behavior with training-time
    # normalization choices from the notebook pipeline.
    hybrid = NepaliHybridTokenizer(subword=False, preprocess=lambda t: t.replace("\u200b", ""))
    tokens = hybrid.tokenize("नेपाल\u200bबाट")

    assert "नेपालबाट" in [t.text for t in tokens]


def test_create_tokenizer_rule_mode_returns_rule_tokenizer():
    tok = create_tokenizer(mode="rule")
    assert isinstance(tok, NepaliTokenizer)


def test_create_tokenizer_hybrid_mode_returns_hybrid_tokenizer_when_available():
    tok = create_tokenizer(mode="hybrid", fallback_to_rule=False)
    assert isinstance(tok, NepaliHybridTokenizer)


def test_create_tokenizer_hybrid_falls_back_to_rule_for_missing_model():
    tok = create_tokenizer(
        mode="hybrid",
        model_path="definitely_missing.model",
        fallback_to_rule=True,
    )
    assert isinstance(tok, NepaliTokenizer)


def test_create_tokenizer_hybrid_strict_mode_raises_for_missing_model():
    with pytest.raises(FileNotFoundError):
        create_tokenizer(
            mode="hybrid",
            model_path="definitely_missing.model",
            fallback_to_rule=False,
        )


def test_create_tokenizer_rejects_invalid_mode():
    with pytest.raises(ValueError):
        create_tokenizer(mode=cast(Any, "invalid"))

if __name__ == "__main__":
    pytest.main([__file__])
