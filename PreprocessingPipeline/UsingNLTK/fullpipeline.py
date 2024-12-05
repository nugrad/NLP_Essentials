import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Example larger text
text = """
Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. 
The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both valuable and meaningful. 
Applications of NLP include machine translation, speech recognition, sentiment analysis, and text summarization, among others.
"""

# Tokenization and Stopword Removal combined
stop_words = set(stopwords.words("english"))
words = [word for word in word_tokenize(text) if word.lower() not in stop_words]

# Lemmatization (instead of stemming, as lemmatization is usually preferred)
lemmatizer = WordNetLemmatizer()
processed_words = [lemmatizer.lemmatize(word) for word in words]
# Final processed text
final_processed_text = " ".join(processed_words)

# Step 6: Display the final processed text
print("Final Processed Text for Training:", final_processed_text)
