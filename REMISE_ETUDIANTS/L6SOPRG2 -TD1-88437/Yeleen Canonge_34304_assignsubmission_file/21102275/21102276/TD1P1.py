#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:17:23 2024

@author: Yeleen
"""

import glob, re, json


def lire_fichier(chemin):
    with open(chemin, encoding="utf-8") as f:
        return f.read()

def extraire_trigrammes(chaine):
    liste_mots = re.findall(r'\b\w+\b', chaine)
    trigrammes = [mot[i:i+3] for mot in liste_mots for i in range(len(mot)-2) if mot.isalpha()]
    return trigrammes

def generer_stats_trigrammes():
    dic_langues = {}

    for chemin in glob.glob("corpus_multi/*/appr/*"):
        chaine = lire_fichier(chemin)
        nom_langue = re.split("/", chemin)[-3][:2]

        trigrammes = extraire_trigrammes(chaine)

        dictri_caracteres = {}
        for trigramme in trigrammes:
            if trigramme not in dictri_caracteres:
                dictri_caracteres[trigramme] = 1
            else:
                dictri_caracteres[trigramme] += 1

        liste_tri_trigrammes = sorted(dictri_caracteres.items(), key=lambda x: x[1], reverse=True)

        dic_langues[nom_langue] = [trigramme for trigramme, _ in liste_tri_trigrammes[:10]]

    with open('trigrammes.json', 'w', encoding='utf-8') as json_file:
        json.dump(dic_langues, json_file, ensure_ascii=False, indent=4)

# Appelle la fonction pour générer les statistiques et sauvegarder en JSON
generer_stats_trigrammes()
