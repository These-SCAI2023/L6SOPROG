#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 08:43:20 2024

@author: ceres
"""

# Diagnostique langue
import glob
import json
from TD1_CKP import *


def lire_json(chemin):
    with open(chemin) as mon_fichier:
        data = json.load(mon_fichier)
    return data

path_mod="./model/"
path_texte="./corpus_multi/*/test/*"

for pth_md in glob.glob(f"{path_mod}/*10.json"):
    # print(pth_md)
    model=lire_json(pth_md)

for pth_txt in glob.glob(path_texte):
    liste_fich=[]
    liste_fich.append(pth_txt)
    langue_txt=lang(pth_txt)
    dic_lang=get_dic_langues(liste_fich)
    dic_mod=get_model(dic_lang,10)
    # print(dic_lang)
    # print(dic_mod)
    for v in dic_mod.values():
        for val in model.values():
        
            compare_mod=set(v).intersection(set(val))
            print(model.keys(),dic_mod.keys())
            print(len(compare_mod))
    
    
    

    
