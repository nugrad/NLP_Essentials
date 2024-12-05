# Use TF-IDF to filter out unimportant words.
# Each row represents a document, and each column represents the TF-IDF score for a unique word.
# High values indicate words important to a specific document but not frequent across the corpus.
# For example, if "cat" appears many times in one document but rarely in others, it will have a high TF-IDF score for that document,
#  indicating its importance.
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import spacy

nlp=spacy.load("en_core_web_sm")

# Sample documents
documents = [
    "The cat sat on the mat.",
    "The dog barked at the cat.",
    "Cats and dogs are friendly animals."
]
# Step 2: Preprocess with SpaCy
def spacy_preprocess(doc):
    spacy_doc = nlp(doc.lower())  # Lowercase and process with SpaCy
    tokens = [
        token.lemma_ for token in spacy_doc  # Lemmatize tokens
        if not token.is_stop and not token.is_punct and not token.is_digit
    ]
    return " ".join(tokens)

# Apply preprocessing to all documents
preprocessed_docs = [spacy_preprocess(doc) for doc in documents]

print("Preprocessed Documents:", preprocessed_docs)
# Step 1: Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Step 2: Fit and transform the documents
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs)

vocabulary=tfidf_vectorizer.get_feature_names_out()

print("Vocabulary:", vocabulary)
matrix_dense=tfidf_matrix.toarray()
print("\nTF-IDF  Matrix:")
print(matrix_dense)

# Create a DataFrame
df = pd.DataFrame(matrix_dense, columns=vocabulary)

print("TF-IDF DataFrame:")
print(df)