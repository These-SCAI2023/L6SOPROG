# -*- coding: utf-8 -*-

#MODULES
import glob
import json
from nltk import ngrams


#FONCTIONS
def lire_fichier(chemin):
    with open(chemin, encoding = 'utf-8') as f:
        chaine = f.read()
    return chaine

#MAIN

liste_fichiers = glob.glob("corpus_multi/*/*/*")
print(liste_fichiers) #La variable est vide sur Spyder alors qu'elle se remplit avec le même programme sur Jupyter
print("Nombre de fichiers : %i"%len(liste_fichiers))
#for chemin in liste_fichiers:
    #print(chemin)
    #print(chemin.split("\\"))
    #1/0
    
dic_langues = {}
liste_fichiers_appr = glob.glob("corpus_multi/*/appr/*")
print("Nombre de fichiers : %i"%len(liste_fichiers_appr))
for chemin in liste_fichiers_appr:
    print(chemin)
    dossiers = chemin.split("\\")
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
    print(dic_langues)
    #1/0
    
dic_modeles = {}
for langue, dic_effectifs in dic_langues.items():
    paires = [[effectif, mot] for mot, effectif in dic_effectifs.items()]
    liste_tri = sorted(paires)[-10:]
    dic_modeles[langue] = [mot for effectif, mot in liste_tri]
    with open("models.json", "w") as w:
        w.write(json.dumps(dic_modeles, indent = 2))

with open("models.json", "r") as f:
    dic_modeles = json.load(f)
    
liste_fichiers_test = glob.glob("corpus_multi/*/test/*")
print("Nombre de fichiers : %i"%len(liste_fichiers_test))
for chemin in liste_fichiers_test:
    dossiers = chemin.split("\\")
    langue = dossiers[1]
    chaine = lire_fichier(chemin)
    mots = chaine.split()
    dic_freq_texte = {}
    for m in mots:
        if m not in dic_freq_texte:
            dic_freq_texte[m] = 1
        else:
            dic_freq_texte[m] += 1
    print(dic_freq_texte)
    1/0
    paires = [[effectif, mot] for mot, effectif in dic_freq_texte.items()]
    liste_tri = sorted(paires)[-10:]
    plus_frequents = set([mot for effectif, mot in liste_tri])
    print(plus_frequents)
    1/0
    print("Document en %s"%langue)
    for langue_ref, model in dic_modeles.items():
        mots_communs = set(model).intersection(plus_frequents)
        print("%i mots en commun avec le modele (%s):"%(len(mots_communs), langue_ref))
        print(mots_communs)
    1/0
    liste_predictions = []
    print("Document en %s"%langue)
    for langue_ref, model in dic_modeles.items():
        mots_communs = set(model).intersection(plus_frequents)
        NB_mots_communs = len(mots_communs)
        liste_predictions.append([NB_mots_communs, langue_ref])
        print(sorted(liste_predictions))
        
#Trigrammes

trigrammes = list(ngrams(dic_modeles.values(), 3))
print(trigrammes)