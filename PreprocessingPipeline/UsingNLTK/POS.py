# POS tagging is the process of assigning a part of speech (like noun, verb, adjective, etc.) to each word in a sentence based on its definition and context.
# It helps in understanding the grammatical structure of a sentence.
import nltk
from nltk import word_tokenize
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')
nltk.download('punkt_tab')


# Example text
text = "The quick brown fox jumps over the lazy dog."

# Tokenize the text
words = word_tokenize(text)

# POS tagging
pos_tags = nltk.pos_tag(words)
print("POS Tags:", pos_tags)
