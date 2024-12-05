from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Define sentences with "bank" in different contexts
sentences = [
    "I went to the bank to deposit money.",  # Finance context
    "The river bank was full of blooming flowers."  # Nature context
]

# Tokenize sentences and encode them into BERT's input format
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# Pass the inputs through BERT to get embeddings
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)

# Get the embeddings for the word "bank" in each sentence
# Find the index of the word "bank" in each tokenized sentence
tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in inputs["input_ids"]]
bank_indices = [tokens[i].index("bank") for i in range(len(sentences))]

# Extract embeddings for "bank"
bank_embeddings = [last_hidden_state[i, idx, :].detach().numpy() for i, idx in enumerate(bank_indices)]

# Print information about embeddings
for i, embedding in enumerate(bank_embeddings):
    print(f"\nContext: {sentences[i]}")
    print(f"Tokenized Sentence: {tokens[i]}")
    print(f"Word 'bank' Embedding (First 10 Dimensions): {embedding[:10]}")
