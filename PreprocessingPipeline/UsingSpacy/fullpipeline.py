import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Example text (larger)
text = """
Natural Language Processing is a fascinating field of Artificial Intelligence. 
It enables computers to understand and generate human language, opening up possibilities in translation, sentiment analysis, and much more. 
With advancements in NLP, tasks that were once considered complex have become accessible to developers worldwide.
"""

# Step 1: Process the text using SpaCy
doc = nlp(text)

# Step 2: Tokenization, Stopword Removal, Punctuation Removal, and Lemmatization
# Final processed text, ready for training
processed_text = " ".join([token.lemma_ for token in doc if not (token.is_stop or token.is_punct)])


# Step 3: Display the final processed text
print("Final Processed Text for Training:", processed_text)
