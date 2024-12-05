import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel
import numpy as np

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Define a small labeled dataset
data = [
    ("I deposited my paycheck at the bank.", "Finance"),
    ("The river bank is a great spot for fishing.", "Nature"),
    ("Apple released the latest version of the iPhone.", "Technology"),
    ("The python programming language is widely used in AI.", "Technology"),
    ("I love the view of the river bank.", "Nature"),
    ("The bank approved my loan application.", "Finance"),
    ("I ate an apple while working on my laptop.", "Nature"),
    ("Python scripts can automate tasks easily.", "Technology"),
]

# Separate sentences and labels
sentences, labels = zip(*data)

# Encode sentences using BERT
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
sentence_embeddings = outputs.pooler_output.detach().numpy()  # Shape: (num_sentences, hidden_size)

# Map labels to numerical values
label_to_idx = {"Finance": 0, "Nature": 1, "Technology": 2}
idx_to_label = {v: k for k, v in label_to_idx.items()}
numerical_labels = np.array([label_to_idx[label] for label in labels])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(sentence_embeddings, numerical_labels, test_size=0.3, random_state=42)

# Train a simple classifier (Logistic Regression)
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = classifier.predict(X_test)
# Evaluate the classifier
print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=idx_to_label.values(),
        labels=list(label_to_idx.values()),  # Explicitly specify all possible labels
    )
)

# Function to classify user input
def classify_sentence(user_sentence):
    # Preprocess and encode the user's input
    input_tokenized = tokenizer(user_sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        input_embedding = model(**input_tokenized).pooler_output.detach().numpy()

    # Predict the category
    prediction = classifier.predict(input_embedding)
    category = idx_to_label[prediction[0]]
    return category

# User input loop
print("\n=== Sentence Classification ===")
while True:
    user_input = input("\nEnter a sentence (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    predicted_category = classify_sentence(user_input)
    print(f"The sentence is classified as: **{predicted_category}**")
