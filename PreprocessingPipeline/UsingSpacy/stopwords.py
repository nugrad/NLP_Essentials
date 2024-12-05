import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Example text
text = "This is a simple example to demonstrate stopword removal in natural language processing."

# Process the text
doc = nlp(text)

# Remove stopwords
filtered_words = [token.text for token in doc if not token.is_stop]
print("Filtered Words (Without Stopwords):", filtered_words)
