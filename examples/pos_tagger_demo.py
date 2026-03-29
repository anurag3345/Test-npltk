from __future__ import annotations

from pathlib import Path
import sys

# Allow running this example directly from a source checkout.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from npltk import POSTagger, create_tokenizer


OUT_PATH = Path("pos_tagger_output.txt")

TEST_INPUTS = [
    "नेपाल एक सुन्दर देश हो।",
    "आज मौसम राम्रो छ र हामी बाहिर जान्छौं।",
    "विद्यालयमा विद्यार्थीहरूले नेपाली भाषा सिक्छन्।",
    "Facebookमा नयाँ फोटो हालें अनि साथीहरूलाई देखाएँ।",
]


def main() -> None:
    tokenizer = create_tokenizer(
        mode="rule",
        split_into_sentences=False,
        keep_punct=True,
    )
    tagger = POSTagger()

    lines: list[str] = []
    lines.append("npltk POS Tagger Demo")
    lines.append("=" * 60)
    lines.append("")

    for idx, text in enumerate(TEST_INPUTS, start=1):
        tokens = tokenizer.tokenize(text)
        token_texts = [t.text for t in tokens]
        tagged = tagger.tag_with_tokens(token_texts)

        lines.append(f"[CASE {idx}]")
        lines.append(f"INPUT  : {text}")
        lines.append(f"TOKENS : {token_texts}")
        lines.append("TAGS   :")

        for tok, tag in tagged:
            lines.append(f"  {tok:<18} -> {tag}")

        lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()