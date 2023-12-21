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

import re
def afficher_contextes(chaine, terme, taille_contexte = 30):
  match = re.search(terme, chaine)
  contexts = []
  while match is not None:
    #Les bornes gauche et droite autour du mot :
    gauche = max(match.start()-1, 0)-taille_contexte
    droite =  match.end()+1+taille_contexte
    contexts.append(chaine[gauche:droite])
    chaine = chaine[match.end():]
    match = re.search(terme, chaine)
  for c in contexts:
    print(c)

import json
f = open("index.json", "r")
index = json.load(f)
f.close()
print(index["Commission"])
1/0
requete = "Commission Européenne"

for mot in requete.split():
  if mot in index:
    for chemin in index[mot]:
      chaine = lire_fichier(chemin)
      afficher_contextes(chaine, mot)
