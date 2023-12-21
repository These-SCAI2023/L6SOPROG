def lire_fichier(chemin):
  f = open(chemin)
  chaine = f.read()
  f.close()
  return chaine

def decouper_en_mots(chaine):
  liste_mots = chaine.split()
  return liste_mots
import os
try:
  os.makedirs("Modeles")
except:
  pass
import glob, re, json

for N in range(1, 21):
  print("Pour NB mots = %i"%N)
  dic_langues = {}
  for chemin in glob.glob("corpus_multi/*/appr/*"):
    dossiers = chemin.split("/")#à adapter
    langue = dossiers[1]
    dic_langues.setdefault(langue, {})#refactor
    chaine = lire_fichier(chemin)
    mots = chaine.split()
    for m in mots:
      dic_langues[langue].setdefault(m, 0) #refactor
      dic_langues[langue][m] += 1
  dic_models = {}
  for langue, dic_effectifs in dic_langues.items():
    paires = [[effectif, mot] for mot, effectif in  dic_effectifs.items()]
    liste_tri = sorted(paires)[-N:]#les N mots fréquents
    dic_models[langue] = [mot for effectif, mot in liste_tri]
  w = open("Modeles/models_NBmots=%i.json"%N, "w")
  w.write(json.dumps(dic_models, indent = 2))
  w.close()
