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
import sklearn
from sklearn.metrics import DistanceMetric 
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy 
import pandas
import seaborn as sns
import matplotlib.pyplot as plt


def lire_fichier(chemin):
    with open(chemin, encoding="utf-8") as f:
        chaine = f.read()
    return chaine

def lire_json(chemin):
    with open(chemin) as mon_fichier:
        data = json.load(mon_fichier)
    return data

def stocker_json(chemin,contenu):
    with open(chemin, "w", encoding="utf-8") as w:
        w.write(json.dumps(contenu, indent =2,ensure_ascii=False))
    return

def lang(chemin):
        dossiers = chemin.split("/")
        langue = dossiers [2]
        return langue

def get_dic_langues(liste_fichiers):
    dic_langues = {}
    for chemin in liste_fichiers:
        langue=lang(chemin)
        chaine = lire_fichier(chemin)
## J'enlève les espaces pour ne pas me retrouver avec des trigrammes qui comptent un espace
        chaine=re.sub(r"\s+", "", chaine)
        if langue not in dic_langues:
            dic_langues[langue] = {}
##Je détermine les trigrammes sur tout le texte
        for i in range(len(chaine)-2):
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

def distance_cos(mod1,mod2):
    V = CountVectorizer(ngram_range=(1,4),analyzer='char')
    X = V.fit_transform([mod1, mod2]).toarray()
    distance_tab1=sklearn.metrics.pairwise.cosine_distances(X)                  
    distance=float(distance_tab1[0][1])
    return distance


    
    