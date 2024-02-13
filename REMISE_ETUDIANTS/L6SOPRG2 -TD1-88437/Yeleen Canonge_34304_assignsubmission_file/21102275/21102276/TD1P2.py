#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:59:34 2024

@author: Yeleen
"""

import glob, re, json

# Définir la fonction pour lire un fichier
def lire_fichier(chemin):
    with open(chemin, encoding="utf-8") as f:
        return f.read()

# Définir la fonction pour extraire les trigrammes
def extraire_trigrammes(chaine):
    liste_mots = re.findall(r'\b\w+\b', chaine)
    trigrammes = set(mot[i:i+3] for mot in liste_mots for i in range(len(mot)-2) if mot.isalpha() and not any(c.isdigit() or c == '/' for c in mot))
    return trigrammes

# Définir la fonction pour estimer la langue en fonction des trigrammes
def estimer_langue(trigrammes, dic_langues):
    score_langues = {}

    for langue, dic_trigrammes in dic_langues.items():
        trigrammes_langue = set(dic_trigrammes)
        intersection = trigrammes_langue.intersection(trigrammes)
        score_langues[langue] = len(intersection)

    langue_estimee = max(score_langues, key=score_langues.get)
    return langue_estimee

# Définir la fonction pour évaluer les résultats
def evaluer(VP, FN, FP):
    if VP != 0:
        rappel = VP / (VP + FN)
        precision = VP / (VP + FP)
        f_mesure = (2 * rappel * precision) / (precision + rappel)
    else:
        rappel, precision, f_mesure = 0, 0, 0
    return rappel, precision, f_mesure

# Charger le dictionnaire des trigrammes par langue à partir du fichier JSON
with open('trigrammes.json', 'r', encoding='utf-8') as json_file:
    dic_langues = json.load(json_file)

# Définir le chemin du dossier contenant les fichiers
chemin_dossier = "corpus_multi/*/test/*"

# Définir les compteurs pour l'évaluation
VP_total = 0
FN_total = 0
FP_total = 0

# Traiter chaque fichier dans le dossier
for chemin in glob.glob(chemin_dossier):
    chaine = lire_fichier(chemin)
    nom_langue = re.split("/", chemin)[-3][:2]

    trigrammes = set(extraire_trigrammes(chaine))

    # Estimer la langue en fonction des trigrammes
    langue_estimee = estimer_langue(trigrammes, dic_langues)

    # Comparer la langue estimée avec la vraie langue
    if nom_langue == langue_estimee:
        VP_total += 1
    else:
        FN_total += 1
        FP_total += 1

    # Afficher la langue du document, la langue estimée, et les trois langues les plus proches
    print(f"Langue du document: {nom_langue}")
    print(f"Langue estimée: {langue_estimee}")

    # Trouver les trois langues les plus proches
    langues_proches = sorted(dic_langues.keys(), key=lambda x: len(set(dic_langues[x]).intersection(trigrammes)), reverse=True)[1:4]
    print(f"Langues les plus proches: {', '.join(langues_proches)}")

    # Calculer le pourcentage de correspondance pour les trois langues les plus proches
    pourcentage_proches = [len(set(dic_langues[langue]).intersection(trigrammes)) / len(dic_langues[langue]) * 100 for langue in langues_proches]
    print(f"Pourcentage de correspondance pour les langues proches: {', '.join([f'{langue}: {pourcentage:.2f}%' for langue, pourcentage in zip(langues_proches, pourcentage_proches)])}\n")

# Évaluer les résultats
rappel, precision, f_mesure = evaluer(VP_total, FN_total, FP_total)
print(f"Rappel: {rappel}")
print(f"Précision: {precision}")
print(f"F-Mesure: {f_mesure}")

# Calculer le score de confiance en pourcentage
pourcentage_vp = (VP_total / (VP_total + FN_total)) * 100

# Afficher le score de confiance en pourcentage
print(f"Score de Confiance: {pourcentage_vp:.2f}%")
