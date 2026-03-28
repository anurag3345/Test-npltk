"""
npltk - Nepali Language Processing Toolkit
-------------------------------------------
Public API focuses on one tokenizer entry-point:
- create_tokenizer
"""
from .tokenizer.factory import create_tokenizer
from .lemmatizer import Lemmatizer

__all__ = ["create_tokenizer", "Lemmatizer"]

__version__ = "0.1.0"
__author__ = [
    "Anurag Sharma",
    "Anita Budha Magar",
    "Apeksha Parajuli",
    "Apeksha Katwal"
]
__credits__ = [
    "Pukar Karki (Project Supervisor)"
]
