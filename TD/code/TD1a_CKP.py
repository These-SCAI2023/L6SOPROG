#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:14:00 2024

@author: ceres
"""
import glob
from TD1_CKP import *

liste_fichiers=[]
c=1
liste_corpus=["appr","test"]
path=f"./corpus_multi/*/{liste_corpus[c]}/*"

for fichier in glob.glob(path):
    liste_fichiers.append(fichier) 
print("Nombre de fichiers : %i"%len(liste_fichiers))


dic_langues = get_dic_langues(liste_fichiers)
print(dic_langues.keys())


for NB_mots in [10, 20, 30, 43]:
    dic_modeles = get_model(dic_langues, NB_mots)
    # stocker_json(f"models/{liste_corpus[c]}_models_3gram_%i.json"%NB_mots,dic_modeles)
