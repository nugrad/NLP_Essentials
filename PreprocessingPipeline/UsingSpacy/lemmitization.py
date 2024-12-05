import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("running runs easily studies better")

# Extract Lemmas
lemmas = [token.lemma_ for token in doc]
print("Lemmatized Words:", lemmas)
