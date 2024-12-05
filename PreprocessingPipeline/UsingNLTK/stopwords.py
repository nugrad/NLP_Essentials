# Stopwords are common words in a language that carry little to no semantic meaning in certain contexts. Examples in English include:

# "the", "is", "in", "and", "to", "a", "of"
# These words are often removed during preprocessing to reduce noise and focus on the more meaningful words.

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Example text
text = "This is a simple example to demonstrate stopword removal in natural language processing."

# Tokenize the text
words = word_tokenize(text)

# Get English stopwords
stop_words = set(stopwords.words("english"))

# Remove stopwords
filtered_words = [word for word in words if word.lower() not in stop_words]
print("Original Words:", words)
print("Filtered Words (Without Stopwords):", filtered_words)
