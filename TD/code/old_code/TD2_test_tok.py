phrase = "L’élision est l’effacement d’une voyelle en fin de mot devant la voyelle commençant le mot suivant."
print(phrase.split())

from mosestokenizer import *
tokenize = MosesTokenizer(lang='fr')
mots = tokenize(phrase)
print(mots)

