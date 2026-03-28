# npltk

Nepali Language Processing Toolkit focused on practical tokenization, normalization, and lemmatization.

Current stable public entry point:
- `create_tokenizer` (top-level API)
- `Lemmatizer` (top-level API)

The project also includes:
- A configurable normalizer pipeline
- Stopword removal utility
- Rule-based and hybrid tokenizer engines under one interface
- Hybrid Nepali lemmatizer pipeline (cache -> dictionary -> rules -> fallback)
- Example scripts and a research notebook for model training/evaluation

## Package Layout

- `src/npltk/__init__.py`: top-level API (`create_tokenizer`)
- `src/npltk/tokenizer/factory.py`: unified tokenizer constructor
- `src/npltk/tokenizer/tokenizer.py`: rule-based tokenizer implementation
- `src/npltk/tokenizer/hybrid_tokenizer.py`: hybrid (rule + SentencePiece) implementation
- `src/npltk/lemmatizer/lemmatizer.py`: primary lemmatizer pipeline
- `src/npltk/lemmatizer/hybrid_lemmatizer.py`: compatibility alias (`HybridLemmatizer`)
- `src/npltk/lemmatizer/rule_stripper.py`: ordered suffix stripping rules
- `src/npltk/lemmatizer/data/lemma_dict.json`: compact exception dictionary
- `src/npltk/normalizer/`: normalization framework and rules
- `src/npltk/stop_word/remover.py`: stopword remover
- `examples/`: runnable examples
- `notebooks/hybrid_tokenizer_research.ipynb`: training/evaluation workflow

## Installation

### Option 1: Editable install (recommended for development)

```bash
pip install -e .
```

### Option 2: Standard install from source directory

```bash
pip install .
```

### Dependency note

Hybrid mode requires `sentencepiece` (already listed in `setup.py`).

## Quick Start

```python
from npltk import create_tokenizer
from npltk import Lemmatizer

# Default: hybrid mode (falls back to rule mode if model/dependency not available)
tokenizer = create_tokenizer()

tokens = tokenizer.tokenize("नेपाल एक सुन्दर देश हो।")
print([t.text for t in tokens])

lemmatizer = Lemmatizer()
print(lemmatizer.lemmatize("गयो"))  # जानु
```

## Public API

## 1) `create_tokenizer`

Location: `src/npltk/tokenizer/factory.py`

Signature:

```python
create_tokenizer(
    *,
    mode: Literal["hybrid", "rule"] = "hybrid",
    split_into_sentences: bool = True,
    keep_punct: bool = True,
    model_path: Optional[str] = None,
    subword: bool = True,
    preprocess: Optional[Callable[[str], str]] = None,
    fallback_to_rule: bool = True,
)
```

Parameters:
- `mode`: tokenizer engine to use
  - `"hybrid"`: hybrid tokenizer (rule + SentencePiece)
  - `"rule"`: pure rule-based tokenizer
- `split_into_sentences`: used by `.tokenize_sentences(...)`
- `keep_punct`: if `True`, punctuation tokens are kept
- `model_path`: optional path to SentencePiece model (`.model`) in hybrid mode
- `subword`: hybrid-only; if `True`, Devanagari word chunks can be split into subwords
- `preprocess`: optional function `(str) -> str` applied before hybrid tokenization
- `fallback_to_rule`: if hybrid initialization fails, fallback to `NepaliTokenizer`

Return type:
- `NepaliTokenizer` or `NepaliHybridTokenizer` (both expose same core methods)

## 2) Tokenizer methods (available on both returned implementations)

### `tokenize(text: str) -> List[Token]`
Returns a flat list of tokens.

### `tokenize_sentences(text: str) -> List[TokenizedSentence]`
Splits into sentences and tokenizes each sentence.

### `detokenize(tokens: List[Token]) -> str`
Reconstructs text from tokens.

## 3) Lemmatizer API

Locations:
- `src/npltk/lemmatizer/lemmatizer.py`
- `src/npltk/lemmatizer/hybrid_lemmatizer.py` (alias)

Classes:
- `Lemmatizer`: primary class
- `HybridLemmatizer`: backward-compatible alias of `Lemmatizer`

Constructor:

```python
Lemmatizer(
  dictionary_path: str | Path | None = None,
  *,
  cache_size: int = 4096,
  min_root_len: int = 2,
)
```

Parameters:
- `dictionary_path`: optional path to custom lemma dictionary JSON
- `cache_size`: maximum in-memory LRU-like cache size
- `min_root_len`: minimum root length for stripping rules

Methods:

```python
lemmatize(word: str) -> str
lemmatize_many(words: list[str]) -> list[str]
```

Lemmatizer flow:
- cache lookup
- dictionary lookup
- rule-based suffix stripping
- fallback to original word

Dictionary formats supported:
- Flat: `{ "word": "lemma" }`
- Compact: `{ "lemma": ["form1", "form2"] }`

Current dictionary is compact and exception-focused.

## Token Data Structures

Location: `src/npltk/tokenizer/types.py`

```python
class TokenType(str, Enum):
    WORD_DEV
    SUBWORD_DEV
    WORD_LAT
    NUM
    PUNCT
    EMOJI
    SYMBOL

@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    type: TokenType

@dataclass
class TokenizedSentence:
    sentence: str
    start: int
    end: int
    tokens: List[Token]
```

Notes:
- `start` and `end` are character offsets (`end` is exclusive).
- In hybrid mode with `preprocess`, spans are relative to preprocessed text.

## Normalizer API

Location: `src/npltk/normalizer/__init__.py`

## 1) `build_normalizer(cfg: NormalizerConfig | None = None) -> Normalizer`
Creates ordered normalization pipeline.

## 2) `Normalizer.normalize(text: str) -> NormResult`
Applies configured rules and returns:
- `NormResult.text`: normalized text
- `NormResult.transforms`: list of applied transforms with metadata

## 3) `NormalizerConfig`

Location: `src/npltk/normalizer/config.py`

Fields (all default `True` unless noted):
- `unicode_nfc`
- `whitespace`
- `invisible_chars`
- `zwj_zwnj`
- `halant_cleanup`
- `diacritic_dedupe`
- `quotation_normalize`
- `digit_normalize`
- `digit_to_nepali` (default `False`)
  - `False`: Nepali digits -> ASCII digits
  - `True`: ASCII digits -> Nepali digits
- `repetition_compress`
- `script_boundary_split`
- `hashtag_mention_split`
- `abbreviation_protect`
- `postposition_split`

### Digit normalization behavior

If you see numeric output like `2026/01/31` instead of `२०२६/०१/३१`, this is expected when `digit_to_nepali=False`.

Example to preserve/display Nepali digits:

```python
from npltk.normalizer import build_normalizer, NormalizerConfig

display_normalizer = build_normalizer(
    NormalizerConfig(digit_normalize=True, digit_to_nepali=True)
)
```

## Stopword API

Location: `src/npltk/stop_word/remover.py`

Class:

```python
StopWordRemover(stopword_file: str | None = None)
```

Method:

```python
remove(tokens: List[Token]) -> Tuple[List[Token], Dict[str, Any]]
```

Returns:
- filtered token list
- info dictionary with:
  - `removed_words`
  - `removed_count`
  - `changed`

## Usage Patterns

### A) Default production-friendly tokenization

```python
from npltk import create_tokenizer

tok = create_tokenizer(mode="hybrid", fallback_to_rule=True)
result = tok.tokenize("घरमा नेपालबाट गएँ।")
print([t.text for t in result])
```

### B) Force rule-only mode

```python
from npltk import create_tokenizer

tok = create_tokenizer(mode="rule")
print([t.text for t in tok.tokenize("Facebookमा photo हाल्नु")])
```

### C) Hybrid with custom model and preprocessing

```python
from npltk import create_tokenizer
from npltk.normalizer import build_normalizer

normalizer = build_normalizer()

hybrid = create_tokenizer(
    mode="hybrid",
    model_path="src/npltk/tokenizer/models/nepali_tokenizer.model",
    preprocess=lambda text: normalizer.normalize(text).text,
    fallback_to_rule=False,
)
```

### D) Tokenize by sentence

```python
from npltk import create_tokenizer

tok = create_tokenizer()
sentences = tok.tokenize_sentences("नेपाल राम्रो छ। school जाउँ!")
for s in sentences:
    print(s.sentence, [t.text for t in s.tokens])
```

### E) Stopword removal after tokenization

```python
from npltk import create_tokenizer
from npltk.stop_word.remover import StopWordRemover

tok = create_tokenizer(mode="rule")
remover = StopWordRemover()

tokens = tok.tokenize("नेपाल सुन्दर देश हो र यहाँ धेरै हिमाल छन् ।")
filtered, info = remover.remove(tokens)
print([t.text for t in filtered])
print(info)
```

### F) Lemmatization (single and batch)

```python
from npltk import Lemmatizer

lem = Lemmatizer()

print(lem.lemmatize("गयो"))       # जानु
print(lem.lemmatize("घरहरूमा"))    # घर

words = ["गयो", "खायो", "किताबबाट", "मलाई"]
print(lem.lemmatize_many(words))
```

### G) Backward compatibility alias

```python
from npltk.lemmatizer.hybrid_lemmatizer import HybridLemmatizer

lem = HybridLemmatizer()
print(lem.lemmatize("गयो"))
```

## Examples Included

- `examples/tokenizer_demo.py`
  - demonstrates rule and hybrid outputs side-by-side
  - writes detailed output to `tokenizer_output.txt`
- `examples/normalize_demo.py`
  - demonstrates each normalization rule and config variant
  - writes output to `normalizer_output.txt`
- `examples/example_stopwords.py`
  - tokenization + stopword removal flow
- `examples/lemmatizer_demo.py`
  - lemmatizer and compatibility alias output side-by-side
  - writes output to `lemmatizer_output.txt`

Run examples from project root:

```bash
python examples/tokenizer_demo.py
python examples/normalize_demo.py
python examples/example_stopwords.py
python examples/lemmatizer_demo.py
```

If needed, set source path:

```bash
PYTHONPATH=src python examples/tokenizer_demo.py
```

## Training and Evaluation Workflow

- Notebook: `notebooks/hybrid_tokenizer_research.ipynb`
- Purpose: model training, experiments, metrics, visual comparisons
- Recommended workflow:
  1. Train/evaluate in notebook
  2. Select best model
  3. Place model in `src/npltk/tokenizer/models/`
  4. Use package API via `create_tokenizer(mode="hybrid", ...)`

## Error Handling Notes

### Hybrid model file not found
- If `fallback_to_rule=True`, tokenizer silently falls back to rule mode.
- If `fallback_to_rule=False`, initialization raises `FileNotFoundError`.

### SentencePiece load mismatch in stubs/runtime
Hybrid loader handles both `Load` and `load` method names internally.

## Current Test Status

Core tokenizer + normalizer tests are passing in this workspace.

## Version

Current package version in source: `0.1.0`
