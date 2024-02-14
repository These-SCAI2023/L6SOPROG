#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:31:28 2024

@author: Yeleen
"""

import glob
import re
import json
import matplotlib.pyplot as plt

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
def estimer_langue(trigrammes, dic_langues, seuil_correspondance=60):
    score_langues = {}

    for langue, dic_trigrammes in dic_langues.items():
        trigrammes_langue = set(dic_trigrammes)
        intersection = trigrammes_langue.intersection(trigrammes)
        pourcentage_correspondance = (len(intersection) / len(dic_trigrammes)) * 100

        if pourcentage_correspondance >= seuil_correspondance:
            score_langues[langue] = pourcentage_correspondance

    if score_langues:
        langue_estimee = max(score_langues, key=score_langues.get)
        return langue_estimee
    else:
        return None

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

# Initialiser le compteur pour les langues proches
resultats_langues_proches = {langue: 0 for langue in dic_langues.keys()}

# Traiter chaque fichier dans le dossier
for chemin in glob.glob(chemin_dossier):
    chaine = lire_fichier(chemin)
    nom_langue = re.split("/", chemin)[-3][:2]

    trigrammes = set(extraire_trigrammes(chaine))

    # Estimer la langue en fonction des trigrammes
    langue_estimee = estimer_langue(trigrammes, dic_langues, seuil_correspondance=60)

    # Comparer la langue estimée avec la vraie langue
    if nom_langue == langue_estimee:
        VP_total += 1
    else:
        FN_total += 1
        FP_total += 1

        # Mettre à jour le compteur pour les langues proches
        if langue_estimee:
            resultats_langues_proches[langue_estimee] += 1

# Afficher la distribution des langues proches
plt.bar(dic_langues.keys(), [resultats_langues_proches[langue] for langue in dic_langues.keys()], color='orange')
plt.xlabel('Langues Proches')
plt.ylabel('Nombre de fois estimées comme proches')
plt.title('Répartition des langues proches estimées (Seuil de 60% de correspondance)')
plt.show()

# Évaluer les résultats
rappel, precision, f_mesure = evaluer(VP_total, FN_total, FP_total)
print(f"Rappel: {rappel}")
print(f"Précision: {precision}")
print(f"F-Mesure: {f_mesure}")

# Calculer le score de confiance en pourcentage
pourcentage_vp = (VP_total / (VP_total + FN_total)) * 100

# Afficher le score de confiance en pourcentage
print(f"Score de Confiance: {pourcentage_vp:.2f}%")
