#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:48:09 2024

@author: ceres
"""
## Générer les modèles de langue
import glob
import json
import re

def lire_fichier(chemin):
    with open(chemin, encoding="utf-8") as f:
        chaine = f.read()
    return chaine

def lang(chemin):
        dossiers = chemin.split("/")
        langue = dossiers [2]
        return langue

def get_dic_langues(liste_fichiers):
    dic_langues = {}
    for chemin in liste_fichiers:
        langue=lang(chemin)
        chaine = lire_fichier(chemin)
        chaine=re.sub(r"\s+", "", chaine)
        if langue not in dic_langues:
            dic_langues[langue] = {}
        for i in range(len(chaine)-3):
            ngram=chaine[i:i+3]
            if ngram not in dic_langues[langue]:
                dic_langues[langue][ngram]=1
            else:
                dic_langues[langue][ngram]+=1
    return dic_langues


def get_model(dic_langues, n_tri):
    dic_modeles = {}
    for langue, dic_effectifs in dic_langues.items():
        paires = []
        for mot, effectif in dic_effectifs.items():
            paires.append([effectif, mot])
        liste_tri = sorted(paires)[-n_tri:]
        dic_modeles[langue]=[mot for effectif, mot in liste_tri]
    return dic_modeles


# liste_fichiers=[]
# c=1
# liste_corpus=["appr","test"]
# path=f"./corpus_multi/*/{liste_corpus[c]}/*"

# for fichier in glob.glob(path):
#     liste_fichiers.append(fichier) 
# print("Nombre de fichiers : %i"%len(liste_fichiers))


# dic_langues = get_dic_langues(liste_fichiers)
# print(dic_langues.keys())


# for NB_mots in [10, 20, 30, 43]:
#     dic_modeles = get_model(dic_langues, NB_mots)
# #     with open(f"models/{liste_corpus[c]}_models_3gram_%i.json"%NB_mots, "w", encoding="utf-8") as w:
# #         w.write(json.dumps(dic_modeles, indent =2,ensure_ascii=False))