from __future__ import annotations

from pathlib import Path

from npltk import Lemmatizer
from npltk.lemmatizer.hybrid_lemmatizer import HybridLemmatizer


OUT_PATH = Path("lemmatizer_output.txt")

TEST_WORDS = [
    "गयो",
    "गए",
    "खायो",
    "घरहरूमा",
    "किताबबाट",
    "गर्दैछन्",
    "मलाई",
    "तिम्रो",
    "उसलाई",
    "अपरिवर्तित",
]


def main() -> None:
    lem = Lemmatizer()
    hybrid_alias = HybridLemmatizer()

    lines: list[str] = []
    lines.append("npltk Lemmatizer Demo")
    lines.append("=" * 60)
    lines.append("")

    for idx, word in enumerate(TEST_WORDS, start=1):
        lemma = lem.lemmatize(word)
        alias_lemma = hybrid_alias.lemmatize(word)
        same = lemma == alias_lemma

        lines.append(f"[CASE {idx}] {word}")
        lines.append(f"  Lemmatizer       : {lemma}")
        lines.append(f"  HybridLemmatizer : {alias_lemma}")
        lines.append(f"  CHECK_ALIAS      : {'PASS' if same else 'FAIL'}")
        lines.append("")

    many = lem.lemmatize_many(TEST_WORDS)
    lines.append("Batch lemmatize_many output:")
    lines.append(str(many))

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
