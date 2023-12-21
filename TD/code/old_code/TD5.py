import re
import glob

def lire_fichier(chemin):
    f = open(chemin, encoding="utf-8")
    chaine = f.read()
    f.close()
    return chaine
def decouper_mots(chaine):
    liste_mots = chaine.split()
    return liste_mots

index = {}

for chemin in glob.glob("corpus_multi/fr/*/*"):
  chaine = lire_fichier(chemin)
  mots = decouper_mots(chaine)
  for mot in mots:
    if mot not in index:
      index[mot] = set()#pour ne pas avoir de doublons
    index[mot].add(chemin)
#print(list(index.keys())[:50])
#print(index["indique"])
for mot in ["indique", "européenne", "toto"]:
  if mot in index:
    print(len(index[mot]))
  else:
    print(0)


import json

index = {mot:list(liste_fichiers) for mot, liste_fichiers in index.items()}

w = open("index.json", "w")
w.write(json.dumps(index))
w.close()
