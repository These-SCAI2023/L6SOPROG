# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict



def lire_fichier(chemin, n):
    
    with open(chemin, encoding="utf-8") as f:
        chaine = f.read()
    return ngram(chaine, n)

def ngram(text, n):
    
    return [text[i:i+n] for i in range(len(text)-n+1)]

def calculate_confidence(liste_pred):
    
    total_scores = sum(score for score, _ in liste_pred)
    if total_scores > 0:
        return [(lang, score / total_scores) for score, lang in liste_pred]
    else:
        return []

def visualize_language_similarity(languages, similarity_matrix):
   
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", xticklabels=languages, yticklabels=languages, cmap="coolwarm")
    plt.title("Language Similarity")
    plt.xlabel("Language")
    plt.ylabel("Language")
    plt.show()


n = 3  
dic_langues = {}
liste_fichiers_appr = glob.glob("corpus_multi/*/appr/*")


for chemin in liste_fichiers_appr:
    dossiers = chemin.split("/")
    langue = dossiers[1]
    if langue not in dic_langues:
        dic_langues[langue] = {}
    ngrams = lire_fichier(chemin, n)
    for ng in ngrams:
        if ng not in dic_langues[langue]:
            dic_langues[langue][ng] = 1
        else:
            dic_langues[langue][ng] += 1


dic_modeles = {}
for langue, dic_effectifs in dic_langues.items():
    paires = [[effectif, ng] for ng, effectif in dic_effectifs.items()]
    liste_tri = sorted(paires)[-10:]
    dic_modeles[langue] = [ng for effectif, ng in liste_tri]


with open("models_ngram.json", "w", encoding='utf-8') as w:
    w.write(json.dumps(dic_modeles, indent=2, ensure_ascii=False))
with open("models_ngram.json", "r", encoding="utf-8") as f:
    dic_modeles = json.load(f)


liste_fichiers_test = glob.glob("corpus_multi/*/test/*")
NB_fichiers = len(liste_fichiers_test)
f = {}
cpt = 0

for chemin in liste_fichiers_test:
    dossiers = chemin.split("/")
    langue = dossiers[1]
    ngrams = lire_fichier(chemin, n)
    dic_freq_texte = {}
    for ng in ngrams:
        if ng not in dic_freq_texte:
            dic_freq_texte[ng] = 1
        else:
            dic_freq_texte[ng] += 1

    paires = [[effectif, ng] for ng, effectif in dic_freq_texte.items()]
    liste_tri = sorted(paires)[-10:]
    plus_frequents = set([ng for effectif, ng in liste_tri])
    liste_pred = []

    for langue_ref, model in dic_modeles.items():
        mots_communs = set(model).intersection(plus_frequents)
        liste_pred.append([len(mots_communs), langue_ref])

    langue_pred = sorted(liste_pred)[-1][1]
    if langue_pred not in f:
        f[langue_pred] = {"vp": 0, "fp": 0, "fn": 0}
    if langue not in f:
        f[langue] = {"vp": 0, "fp": 0, "fn": 0}
        
    if langue_pred == langue:
        cpt += 1
        f[langue]['vp'] += 1
    else:
        f[langue_pred]['fp'] += 1
        f[langue]['fn'] += 1

print("Bonnes prédictions :", cpt)
print("sur", NB_fichiers, "fichiers")
print("La proportion de bonnes réponses:", f"{cpt/NB_fichiers:.2f}")

for langue in f:
    vp = f[langue]['vp']
    fp = f[langue]['fp']
    fn = f[langue]['fn']
    rappel = vp / (vp + fn) if (vp + fn) > 0 else 0
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0
    f1_mesure = 2 * rappel * precision / (precision + rappel) if (precision + rappel) > 0 else 0

    f[langue]['rappel'] = rappel
    f[langue]['precision'] = precision
    f[langue]['f1_mesure'] = f1_mesure

    print(f"Langue: {langue}")
    print(f"  VP: {vp}, FP: {fp}, FN: {fn}")
    print(f"  Rappel (Recall): {rappel:.4f}")
    print(f"  Précision (Precision): {precision:.4f}")
    print(f"  F1 Mesure: {f1_mesure:.4f}")
    print()


languages = list(dic_modeles.keys())
similarity_matrix = np.zeros((len(languages), len(languages)))

for i, lang1 in enumerate(languages):
    for j, lang2 in enumerate(languages):
        if i == j:
            similarity_matrix[i][j] = 1.0
        else:
            set1 = set(dic_modeles[lang1])
            set2 = set(dic_modeles[lang2])
            common_elements = len(set1.intersection(set2))
            total_elements = len(set1.union(set2))
            similarity_matrix[i][j] = common_elements / total_elements if total_elements > 0 else 0

visualize_language_similarity(languages, similarity_matrix)