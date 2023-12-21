def lire_fichier(chemin):
  f = open(chemin)
  chaine = f.read()
  f.close()
  return chaine

def decouper_en_mots(chaine):
  liste_mots = chaine.split()
  return liste_mots

import glob, re

import json
f = open("models.json", "r")
dic_models = json.load(f)
f.close()

liste_langues = dic_models.keys() 
dic_resultats ={lg: {"VP":0, "FP":0, "FN":0} for lg in liste_langues}
erreurs= {}
for chemin in glob.glob("corpus_multi/*/test/*"):
  dossiers = chemin.split("/")
  langue = dossiers[1]
  chaine = lire_fichier(chemin)
  mots = chaine.split()
  dic_freq_texte = {}
  for m in mots:
    if m not in dic_freq_texte:
      dic_freq_texte[m] = 1
    else:
      dic_freq_texte[m] += 1
  paires = [[effectif, mot] for mot, effectif in  dic_freq_texte.items()]
  liste_tri = sorted(paires)[-10:]#les 10 mots fréquents
  plus_frequents = set([mot for effectif, mot in liste_tri])
  liste_predictions = []
  for langue_ref, model in dic_models.items():
    mots_communs = set(model).intersection(plus_frequents)
    NB_mots_communs = len(mots_communs)
    liste_predictions.append([NB_mots_communs, langue_ref])
  liste_predictions = sorted(liste_predictions)
  lg_pred = liste_predictions[-1][1]
  if lg_pred == langue:
    dic_resultats[langue]["VP"]+=1
  else:
    dic_resultats[langue]["FN"]+=1
    dic_resultats[lg_pred]["FP"]+=1
    erreurs.setdefault((langue, lg_pred), 0)
    erreurs[(langue, lg_pred)] +=1

for langue, infos in dic_resultats.items():
  VP = infos["VP"]
  FP = infos["FP"]
  FN = infos["FN"]
  if VP!=0:
    rappel = VP/(VP+FN)
    precision=VP/(VP+FP)
    f_mesure = (2*rappel*precision)/(precision+rappel)
  else:
    rappel, precision, f_mesure = 0, 0, 0
  print("%s : rappel =%f, précision=%f et f-mesure=%f"%(langue,rappel, precision, f_mesure))

for paire, NB_erreurs in erreurs.items():
  print("Pour la paire %s : %i erreurs"%(str(paire), NB_erreurs))
