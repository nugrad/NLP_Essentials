# Both stemming and lemmatization are text normalization techniques used in NLP to reduce words to their root forms.
import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

# Example text
words = ["running", "runner", "ran", "easily", "studies", "studying"]

# Porter Stemmer
porter = PorterStemmer()
stemmed_porter = [porter.stem(word) for word in words]
print("Porter Stemmer:", stemmed_porter)

# Snowball Stemmer
snowball = SnowballStemmer("english")
stemmed_snowball = [snowball.stem(word) for word in words]
print("Snowball Stemmer:", stemmed_snowball)

# Lancaster Stemmer
lancaster = LancasterStemmer()
stemmed_lancaster = [lancaster.stem(word) for word in words]
print("Lancaster Stemmer:", stemmed_lancaster)
