#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:28:10 2024

@author: ceres
"""
#____________TOUS LES IMPORTS
import json
import glob
#____________FIN DE TOUS LES IMPORTS

#____________TOUTES LES FONCTIONS
def lire_fichier (chemin):
    with open(chemin) as json_data: 
        dist =json.load(json_data)
        
        return dist
#____________FIN DE TOUTES LES FONCTIONS    

#____________MAIN   
path_corpora = "./corpora/corpus_eval/*/*/*"
# dans "corpora" un subcorpus = toutes les versions 'un texte'
liste_EN_ocr=[]
liste_EN_pp=[]
for subcorpus in sorted(glob.glob(path_corpora)):

    for path in sorted(glob.glob("%s/*.json"%subcorpus)):    
        texte = lire_fichier(path)

