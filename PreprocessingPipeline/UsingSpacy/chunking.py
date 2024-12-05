import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Example text
text = "The quick brown fox jumps over the lazy dog."

# Process the text
doc = nlp(text)

# Extract noun chunks
for chunk in doc.noun_chunks:
    print(f"Noun Chunk: {chunk.text}")
