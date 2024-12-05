# Import necessary libraries
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

# Load SpaCy for preprocessing
nlp = spacy.load("en_core_web_sm")
corpus = [
    "The cat sat on the mat.",
    "The dog barked at the cat.",
    "Cats and dogs are friendly animals.",
    "I love my cat.",
    "The mat is where the dog sits."
]

# Step 1: Preprocessing using SpaCy
def preprocess_text(text):
    doc = nlp(text.lower())  # Lowercase text and tokenize
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]  # Lemmatize and remove stopwords/punctuation
    return tokens

# Apply preprocessing to the corpus
#This results in cleaner, more meaningful tokens that can be used in vectorization.
preprocessed_corpus = [preprocess_text(sentence) for sentence in corpus]
print("Preprocessed Corpus:", preprocessed_corpus)

# Step 2: Bag of Words (BoW) representation
# This allows the model to understand the frequency of words in each document.
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform([" ".join(tokens) for tokens in preprocessed_corpus])
print("\nBoW Feature Names:", bow_vectorizer.get_feature_names_out())
print("BoW Matrix (Dense Format):\n", bow_matrix.toarray())

# Step 3: TF-IDF representation
#The resulting TF-IDF matrix assigns more weight to words that are unique to a document and less to 
# common words across the entire corpus.
#For example, if "cat" appears many times in one document but rarely in others, it will have a high TF-IDF score for that document,
#indicating its importance.
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(tokens) for tokens in preprocessed_corpus])
print("\nTF-IDF Feature Names:", tfidf_vectorizer.get_feature_names_out())
print("TF-IDF Matrix (Dense Format):\n", tfidf_matrix.toarray())

# Step 4: Word2Vec Embeddings
# In Word2Vec, words with similar meanings tend to have similar vector representations.
# Train a custom Word2Vec model on the preprocessed corpus
word2vec_model = Word2Vec(
    sentences=preprocessed_corpus,  # Preprocessed tokenized sentences
    vector_size=100,  # Dimensionality of the word vectors
    window=5,  # Context window size
    min_count=1,  # Minimum word frequency
    sg=0  # CBOW (Continuous Bag of Words); use sg=1 for Skip-Gram
)

# Function to compute document embeddings by averaging word vectors
#This results in a fixed-length vector that represents the semantic content of each document.
def get_document_vector(tokens, word2vec_model):
    vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)  # Handle cases with no valid tokens

# Compute document embeddings
document_embeddings = [get_document_vector(tokens, word2vec_model) for tokens in preprocessed_corpus]
#The output is a list of document embeddings, where each document is represented by a 100-dimensional vector (since vector_size=100).

print("\nDocument Embeddings Shape:", len(document_embeddings), len(document_embeddings[0]))
print("Sample Document Embedding (First Document):\n", document_embeddings[0])

# Summary of Outputs
print("\nSummary of Outputs:")
print("- Preprocessed Corpus:", preprocessed_corpus)
print("- BoW Matrix Shape:", bow_matrix.shape)
print("- TF-IDF Matrix Shape:", tfidf_matrix.shape)
print("- Document Embeddings Shape:", len(document_embeddings), "x", len(document_embeddings[0]))


# Let's say you have a dataset of customer reviews, and you want to:

# Classify reviews as positive or negative.
# Find similar reviews based on content (for example, to suggest similar reviews to users).
# In this case:

# You would preprocess and vectorize the text using techniques like BoW, TF-IDF, or Word2Vec.
# The Word2Vec document embeddings would then be used to represent each review as a dense vector.
# These embeddings could be fed into a machine learning model (e.g., logistic regression) for classification, or you could use them 
# to calculate similarity between reviews using cosine similarity.