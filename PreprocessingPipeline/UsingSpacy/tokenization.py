import spacy
# load a pre-trained language model in SpaCy.
nlp = spacy.load("en_core_web_sm")
text = "SpaCy is faster, but NLTK is highly customizable. Which one should we choose?"
doc = nlp(text)

# Word Tokenization
words_spacy = [token.text for token in doc]
print("Words (SpaCy):", words_spacy)

# Sentence Tokenization
sentences_spacy = [sent.text for sent in doc.sents]
print("Sentences (SpaCy):", sentences_spacy)
