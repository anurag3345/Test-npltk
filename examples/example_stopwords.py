from npltk.tokenizer.tokenizer import NepaliTokenizer
from npltk.stop_word.remover import StopWordRemover

def main():
    # 1. Initialize the tokenizer
    tokenizer = NepaliTokenizer()
    
    # 2. Initialize the stopword remover
    # It automatically loads the default Nepali stopwords included in the package
    stopword_remover = StopWordRemover()
    
    # Sample text with some common stopwords like "र" (and), "छ" (is), "पनि" (also)
    text = "नेपाल सुन्दर देश हो र यहाँ धेरै हिमालहरु छन् । म पनि नेपाल जान्छु ।"
    
    print(f"Original Text: {text}")
    print("-" * 50)
    
    # 3. Tokenize the text
    tokens = tokenizer.tokenize(text)
    
    print("Tokens Before Stopword Removal:")
    print([t.text for t in tokens])
    print("-" * 50)
    
    # 4. Remove stopwords
    filtered_tokens, info = stopword_remover.remove(tokens)
    
    print("Tokens After Stopword Removal:")
    print([t.text for t in filtered_tokens])
    print("-" * 50)
    
    # 5. Information about what was removed
    print("Stopword Removal Info:")
    print(f"Removed Words: {info['removed_words']}")
    print(f"Count: {info['removed_count']}")
    
    # 6. Detokenize back to string (optional)
    cleaned_text = tokenizer.detokenize(filtered_tokens)
    print("-" * 50)
    print(f"Cleaned Text: {cleaned_text}")

if __name__ == "__main__":
    main()
