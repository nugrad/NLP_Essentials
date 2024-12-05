# Word embeddings are a powerful representation of words in a continuous vector space where 
# semantically similar words are close to each other.
# embeddings capture the context and semantic relationships between words.
# we will use Word2vec
# CBOW (Continuous Bag of Words): Predicts the target word from its surrounding context words.
# Skip-gram: Predicts context words from a given target word.
#in our below  Word2Vec instantiation did not explicitly specify the sg parameter which means,
# the default value is sg=0, meaning CBOW is used.
#if you want to use Skip-gram, modify the code to include sg=1:
from gensim.models import Word2Vec
import spacy
nlp = spacy.load("en_core_web_sm")

# Sample corpus
corpus = [
    "The cat sat on the mat.",
    "The dog barked at the cat.",
    "Cats and dogs are friendly animals.",
    "I love my cat.",
    "The mat is where the dog sits."
]

# Preprocess the corpus
def preprocess_text(doc):
    spacy_doc = nlp(doc.lower())  # Lowercase and process with SpaCy
    tokens = [
        token.lemma_ for token in spacy_doc  # Lemmatize tokens
        if not token.is_stop and not token.is_punct and not token.is_digit
    ]
    return tokens  # Return a list of tokens, not a string

preprocessed_corpus = [preprocess_text(doc) for doc in corpus]
print("Preprocessed Corpus:", preprocessed_corpus)

# Train Word2Vec model
model = Word2Vec(sentences=preprocessed_corpus, vector_size=50, window=5, min_count=1, workers=4)

# Save and load the model
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")

# Explore the model
print("Vocabulary:", model.wv.index_to_key)
print("Vector for 'cat':", model.wv['cat'])
print("Most similar to 'cat':", model.wv.most_similar('cat'))



