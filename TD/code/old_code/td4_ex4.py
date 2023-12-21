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

for taille in range(1, 5):
  print("Pour taille = %i"%taille)
  if os.path.exists("Modeles/models_taille_chaine=%i.json"):
    continue
  dic_langues = {}
  for chemin in glob.glob("corpus_multi/*/appr/*"):
    dossiers = chemin.split("/")#à adapter
    langue = dossiers[1]
    dic_langues.setdefault(langue, {})#refactor
    chaine = lire_fichier(chemin)
    for position in range(len(chaine)):
      m = chaine[position:position+1+taille]
      dic_langues[langue].setdefault(m, 0) #refactor
      dic_langues[langue][m] += 1
  dic_models = {}
  for langue, dic_effectifs in dic_langues.items():
    paires = [[effectif, mot] for mot, effectif in  dic_effectifs.items()]
    liste_tri = sorted(paires)[-10:]#les 10 chaînes plus fréquentes
    dic_models[langue] = [mot for effectif, mot in liste_tri]
  w = open("Modeles/models_taille_chaine=%i.json"%taille, "w")
  w.write(json.dumps(dic_models, indent = 2))
  w.close()

dic_courbes = {}#pour faire des courbes ultérieurement
for taille in range(1, 21):
  print("Pour taille = %i"%taille)
  f = open("Modeles/models_taille_chaine=%i.json"%taille, "r")
  dic_models = json.load(f)
  f.close()
  liste_langues = dic_models.keys() 
  dic_courbes[N] = {lg : [[], [], []] for lg in liste_langues} 
  dic_resultats ={lg: {"VP":0, "FP":0, "FN":0} for lg in liste_langues}
  for chemin in glob.glob("corpus_multi/*/test/*"):
    dossiers = chemin.split("/")
    langue = dossiers[1]
    chaine = lire_fichier(chemin)
    dic_freq_texte = {}
    for position in range(len(chaine)):
      m = chaine[position:position+1+taille]
      dic_freq_texte.setdefault(m, 0) #refactor
      dic_freq_texte[m] += 1
    paires = [[effectif, mot] for mot, effectif in  dic_freq_texte.items()]
    liste_tri = sorted(paires)[-10:]#les 10 chaînes les plus fréquentes
    plus_frequents = set([mot for effectif, mot in liste_tri])
    liste_predictions = []
    for langue_ref, model in dic_models.items():
      mots_communs = set(model).intersection(plus_frequents)
      liste_predictions.append([len(mots_communs), langue_ref])#refactor
    lg_pred = sorted(liste_predictions)[-1][1]#refactor
    if lg_pred == langue:
      dic_resultats[langue]["VP"]+=1
    else:
      dic_resultats[langue]["FN"]+=1
      dic_resultats[lg_pred]["FP"]+=1
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
    #print("%s : rappel =%f, précision=%f et f-mesure=%f"%(langue,rappel, precision, f_mesure))
    dic_courbes[N][langue][0].append(rappel)
    dic_courbes[N][langue][1].append(precision)
    dic_courbes[N][langue][2].append(f_mesure)

w = open("resultats_chaines.json", "w")
w.write(json.dumps(dic_courbes, indent = 2))
w.close()
    
