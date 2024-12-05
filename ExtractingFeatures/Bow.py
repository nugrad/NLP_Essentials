import spacy
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

nlp=spacy.load("en_core_web_sm")
# Sample documents
documents = [
    "The cat sat on the mat.",
    "Dogs are barking loudly outside.",
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

# Step 3: Convert to BoW using CountVectorizer
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(preprocessed_docs)

# Step 4: Output Vocabulary and Matrix
vocabulary=vectorizer.get_feature_names_out()
print("Vocabulary:", vocabulary)
matrix_dense=bow_matrix.toarray()
print("\nBag of Words Matrix:")
print(matrix_dense)


# Create a DataFrame
df = pd.DataFrame(matrix_dense, columns=vocabulary)

print("Bag of Words DataFrame:")
print(df)