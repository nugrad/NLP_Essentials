from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from scipy.spatial.distance import cosine

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Define a small corpus with words having different meanings
corpus = [
    "I went to the bank to deposit my paycheck.",  # Finance context
    "The river bank is a beautiful place to relax.",  # Nature context
    "Apple is a leading tech company based in California.",  # Tech company context
    "I ate a fresh apple for breakfast.",  # Fruit context
    "Python is a popular programming language.",  # Programming context
    "The python slithered through the dense jungle."  # Animal context
]

# Tokenize sentences and encode them into BERT's input format
inputs = tokenizer(corpus, return_tensors="pt", padding=True, truncation=True)

# Pass the inputs through BERT to get embeddings
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)

# Extract contextual embeddings for the target words in each sentence
target_words = ["bank", "bank", "apple", "apple", "python", "python"]
word_embeddings = []

for i, sentence in enumerate(corpus):
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][i])
    target_word_index = tokens.index(target_words[i])  # Find the index of the target word
    embedding = last_hidden_state[i, target_word_index, :].detach().numpy()  # Extract embedding
    word_embeddings.append(embedding)
    print(f"\nSentence: {sentence}")
    print(f"Tokenized: {tokens}")
    print(f"Word '{target_words[i]}' Embedding (First 10 Dimensions): {embedding[:10]}")

# Compare embeddings using cosine similarity
def compare_embeddings(embedding1, embedding2, word1, word2):
    similarity = 1 - cosine(embedding1, embedding2)
    print(f"\nCosine Similarity between '{word1}' and '{word2}': {similarity:.4f}")

# Comparing "bank" in finance vs. nature
compare_embeddings(word_embeddings[0], word_embeddings[1], "bank (finance)", "bank (nature)")

# Comparing "apple" as a company vs. a fruit
compare_embeddings(word_embeddings[2], word_embeddings[3], "apple (company)", "apple (fruit)")

# Comparing "python" as a language vs. an animal
compare_embeddings(word_embeddings[4], word_embeddings[5], "python (language)", "python (animal)")
