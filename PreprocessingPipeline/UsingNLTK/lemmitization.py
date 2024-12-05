from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# Example text
# The second element in each tuple (e.g., "v", "a", "n") represents the part of speech (POS) tag for the word.
words = [("running", "v"), ("better", "a"), ("studies", "n")]

lemmatizer = WordNetLemmatizer()

# Lemmatize words with POS
lemmatized = [lemmatizer.lemmatize(word, pos) for word, pos in words]
print("Lemmatized Words:", lemmatized)
