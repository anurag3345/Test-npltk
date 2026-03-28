import os
import sys

# Add src folder to path for local run
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from npltk.lemmatizer import Lemmatizer


def test_hybrid_lemmatizer_required_examples():
    lem = Lemmatizer()

    assert lem.lemmatize("गयो") == "जानु"
    assert lem.lemmatize("घरहरूमा") == "घर"
    assert lem.lemmatize("खायो") == "खानु"
    assert lem.lemmatize("किताबबाट") == "किताब"


def test_multi_suffix_stripping():
    lem = Lemmatizer()
    assert lem.lemmatize("घरहरूबाट") == "घर"
    assert lem.lemmatize("मान्छेहरूमा") == "मान्छे"


def test_verb_rule_case():
    lem = Lemmatizer()
    assert lem.lemmatize("गर्दैछन्") == "गर्नु"


def test_fallback_behavior():
    lem = Lemmatizer()
    assert lem.lemmatize("अपरिवर्तित") == "अपरिवर्तित"


def test_cache_behavior():
    lem = Lemmatizer(cache_size=2)

    a1 = lem.lemmatize("गयो")
    a2 = lem.lemmatize("गयो")

    assert a1 == "जानु"
    assert a2 == "जानु"
    assert "गयो" in lem.cache
