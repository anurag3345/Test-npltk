"""
normalize_demo.py
-----------------
Demonstrates all normalization rules in the improved npltk normalizer.
"""
from npltk.normalizer import build_normalizer, NormalizerConfig
from pathlib import Path


def main() -> None:
    normalizer = build_normalizer()

    # ── Test cases covering every rule ───────────────────────────────────
    test_cases = {
        "Whitespace + invisible chars":
            "घरमा  \u200B नेपाल\u200Dबाट  गएँ",

        "Halant + diacritic issues":
            "विद्् यार्थी  संंस्कृति",

        "Fancy quotation marks":
            "\u201Cराम्रो\u201D भन्नुभयो \u2018हो\u2019",

        "Nepali digit normalization":
            "मिति २०२६/०१/३१ हो",

        "Social media repetition":
            "हाहाहाहाहा कति राम्रोोोो!!!!!",

        "Script-boundary (mixed lang)":
            "Facebookमा photo हाल्नु lockdownको असर",

        "Hashtag / mention":
            "#NepaliPride trending @username लाई",

        "Abbreviation protection":
            "डा. शर्माले भन्नुभयो। श्री. राम आउनुभयो।",

        "Postposition split":
            "नेपालबाट विदेशसम्म घरहरूमा मान्छेलाई",
    }

    # ── Run and display ──────────────────────────────────────────────────
    lines = []
    lines.append("npltk Normalizer Demo — All Rules")
    lines.append("=" * 55)

    for label, text in test_cases.items():
        result = normalizer.normalize(text)
        lines.append(f"\n[{label}]")
        lines.append(f"  IN : {text}")
        lines.append(f"  OUT: {result.text}")
        if result.transforms:
            for t in result.transforms:
                lines.append(f"    → {t.rule}: {t.meta}")
        else:
            lines.append("    (no changes)")

    output = "\n".join(lines)
    print(output)

    # ── Save to file ─────────────────────────────────────────────────────
    root_dir = Path(__file__).resolve().parents[1]
    output_file = root_dir / "normalizer_output.txt"
    output_file.write_text(output, encoding="utf-8")
    print(f"\nSaved to: {output_file}")


    # ── Demo: config toggling ────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Config example: digits kept as Nepali\n")
    cfg = NormalizerConfig(digit_normalize=True, digit_to_nepali=True)
    norm2 = build_normalizer(cfg)
    r = norm2.normalize("Year 2026 मा 500 रुपैयाँ")
    print(f"  IN : Year 2026 मा 500 रुपैयाँ")
    print(f"  OUT: {r.text}")


if __name__ == "__main__":
    main()
