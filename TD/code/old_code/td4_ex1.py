def lire_fichier(chemin):
  f = open(chemin)
  chaine = f.read()
  f.close()
  return chaine

def decouper_en_mots(chaine):
  liste_mots = chaine.split()
  return liste_mots

import glob, re

dic_langues = {}
for chemin in glob.glob("corpus_multi/*/*/*"):
  dossiers = chemin.split("/")#Sur Linux/Mac: chemin.split("/")
  langue = dossiers[1]
  if langue not in dic_langues:
    dic_langues[langue] = {}
  chaine = lire_fichier(chemin)
  mots = chaine.split()
  for m in mots:
    if m not in dic_langues[langue]:
      dic_langues[langue][m] = 1
    else:
      dic_langues[langue][m] += 1

dic_models = {}
for langue, dic_effectifs in dic_langues.items():
  paires = [[effectif, mot] for mot, effectif in  dic_effectifs.items()]
  liste_tri = sorted(paires)[-10:]#les 10 mots fréquents
  dic_models[langue] = [mot for effectif, mot in liste_tri]

import json
w = open("models.json", "w")
w.write(json.dumps(dic_models, indent = 2))
w.close()
