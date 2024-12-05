import nltk
from nltk.tokenize import word_tokenize,sent_tokenize

nltk.download('punkt_tab')
nltk.download('punkt')

text = "SpaCy is faster, but NLTK is highly customizable. Which one should we choose?"

# Word Tokenization
words = word_tokenize(text)
print("Words (NLTK):", words)

# Sentence Tokenization
sentences = sent_tokenize(text)
print("Sentences (NLTK):", sentences)