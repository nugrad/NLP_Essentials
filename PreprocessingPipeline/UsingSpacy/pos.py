import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Example text
text = "The quick brown fox jumps over the lazy dog."

# Process the text
doc = nlp(text)

# Extract POS tags
for token in doc:
    print(f"{token.text}: {token.pos_} ({token.tag_})")
