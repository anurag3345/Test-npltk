from __future__ import annotations

from pathlib import Path

from npltk.normalizer import NormalizerConfig, build_normalizer
from npltk.tokenizer import create_tokenizer


OUT_PATH = Path("tokenizer_output.txt")

TEST_INPUTS = [
    "नेपाल एक सुन्दर देश हो।",
    "घरमा नेपालबाट गएँ। अनि school गएँ!",
    "मिति २०२६/०१/३१ हो 🙂",
    "Facebookमा photo हाल्नु lockdownको असर",
    "डा. शर्माले भन्नुभयो। श्री. राम आउनुभयो।",
]


def main() -> None:
    # Internal form: stable normalization for tokenization/evaluation.
    normalizer = build_normalizer(
        NormalizerConfig(digit_normalize=True, digit_to_nepali=False)
    )
    # Display form: convert digits back to Nepali for user-facing output.
    display_normalizer = build_normalizer(
        NormalizerConfig(digit_normalize=True, digit_to_nepali=True)
    )
    # One API: default tokenizer is hybrid (falls back to rule tokenizer
    # automatically if model/dependency is unavailable).
    hybrid = create_tokenizer(
        mode="hybrid",
        split_into_sentences=True,
        keep_punct=True,
        subword=True,
        fallback_to_rule=True,
    )
    tokenizer = create_tokenizer(
        mode="rule",
        split_into_sentences=True,
        keep_punct=True,
    )

    lines = []
    lines.append("Tokenizer Demo Output")
    lines.append("=" * 70)
    lines.append("")

    for case_idx, text in enumerate(TEST_INPUTS, start=1):
        norm_result = normalizer.normalize(text)
        norm_text = norm_result.text

        flat_tokens = tokenizer.tokenize(norm_text)
        flat_recon = tokenizer.detokenize(flat_tokens)
        display_recon = display_normalizer.normalize(flat_recon).text
        recon_ok = flat_recon == norm_text

        sent_groups = tokenizer.tokenize_sentences(norm_text)
        span_ok = all(norm_text[s.start:s.end] == s.sentence for s in sent_groups)

        lines.append(f"[CASE {case_idx}]")
        lines.append(f"IN          : {text}")
        lines.append(f"NORMALIZED  : {norm_text}")
        lines.append(f"RECONSTRUCT : {display_recon}")
        lines.append(f"CHECK_RECON : {'PASS' if recon_ok else 'FAIL'}")
        lines.append(f"CHECK_SPANS : {'PASS' if span_ok else 'FAIL'}")
        lines.append(f"TOKENS_FLAT : {len(flat_tokens)}")
        lines.append(f"SENT_COUNT  : {len(sent_groups)}")
        lines.append(f"TOKENS_LIST : {[t.text for t in flat_tokens]}")
        lines.append("")

        hybrid_tokens = hybrid.tokenize(norm_text)
        hybrid_recon = hybrid.detokenize(hybrid_tokens)
        hybrid_display_recon = display_normalizer.normalize(hybrid_recon).text
        hybrid_sent_groups = hybrid.tokenize_sentences(norm_text)
        lines.append(f"HYBRID_RECON : {hybrid_display_recon}")
        lines.append(f"HYBRID_COUNT : {len(hybrid_tokens)}")
        lines.append(f"HYBRID_LIST  : {[t.text for t in hybrid_tokens]}")
        lines.append("")

        for sent_idx, sent in enumerate(hybrid_sent_groups, start=1):
            sent_token_texts = [t.text for t in sent.tokens]
            lines.append(f"  [HYBRID SENT {sent_idx}] {sent.sentence}")
            lines.append(f"    TOKENS: {sent_token_texts}")
            lines.append("")

        for sent_idx, sent in enumerate(sent_groups, start=1):
            sent_token_texts = [t.text for t in sent.tokens]
            lines.append(f"  [SENT {sent_idx}] {sent.sentence}")
            lines.append(f"    TOKENS: {sent_token_texts}")
            lines.append("")

        lines.append("-" * 70)
        lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
