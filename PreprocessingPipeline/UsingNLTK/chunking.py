# Chunking, also called shallow parsing, is the process of grouping POS-tagged words into higher-level phrases, such as noun phrases (NP), verb phrases (VP), and prepositional phrases (PP).
# For example:
# Sentence: "The quick brown fox jumps over the lazy dog."
# Chunk: "[NP The quick brown fox] [VP jumps] [PP over [NP the lazy dog]]."
import nltk
from nltk import word_tokenize, pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# Example text
text = "The quick brown fox jumps over the lazy dog."

# Tokenize and POS tag
words = word_tokenize(text)
pos_tags = pos_tag(words)

# Define a chunking grammar (e.g., for noun phrases)
chunk_grammar = r"""
    NP: {<DT>?<JJ>*<NN>}  # Noun phrase: optional determiner, adjectives, noun
"""

# Create a chunk parser
chunk_parser = nltk.RegexpParser(chunk_grammar)

# Parse the POS-tagged text into chunks
tree = chunk_parser.parse(pos_tags)
print(tree)
tree.draw()
