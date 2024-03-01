#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:42:07 2024

@author: ceres
"""
from TD2_CKP import *



path_data="../ressources/ressources_TD1_Entite-nommee/Texte/*/*/"
for chemin in glob.glob(path_data):
    for chemin_fichier in glob.glob("%s*token*"%chemin):
        output=chemin_fichier.split("/")[4]
        path_output=output+"_zipf_corpus.png"
        print(chemin_fichier)
        data=lire_fichier_json(chemin_fichier)
        print("Quantité des mots (tokens) :", len(data))
        print("Quantité des mots differentes (types) :", len(set(data)))
        texte_dict = texte_to_dict(data)
        texte_list=dict_to_list(texte_dict)
        affichage=afficher_n(texte_list, 15)
        if "split" in chemin_fichier:
            texte_list1 = texte_list
            
    
        else:
            texte_list2 = texte_list
    print(texte_list1)
    
    plot_zipf(texte_list1[:],texte_list2[:],chemin,path_output, log=True)
    