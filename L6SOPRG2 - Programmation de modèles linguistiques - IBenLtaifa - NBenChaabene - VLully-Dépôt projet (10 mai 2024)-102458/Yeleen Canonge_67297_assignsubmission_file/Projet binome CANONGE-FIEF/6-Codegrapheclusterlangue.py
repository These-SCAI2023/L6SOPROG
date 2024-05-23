#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:23:19 2024

@author: Yeleen
"""

import json
import matplotlib.pyplot as plt
import os

# Chargement des données depuis le fichier JSON
with open("clusters_hierarchiqueB.json") as json_file:
    donnees_B = json.load(json_file)

with open("clusters_hierarchiqueT.json") as json_file:
    donnees_T = json.load(json_file)

with open("clusters_hierarchique4.json") as json_file:
    donnees_4 = json.load(json_file)

with open("clusters_hierarchique5.json") as json_file:
    donnees_5 = json.load(json_file)

# Récupération du nombre de clusters par langue pour chaque type de fichier
langues = list(donnees_B.keys())
nombre_clusters_B = [donnees_B[langue]['nombre_clusters'] for langue in langues]
nombre_clusters_T = [donnees_T[langue]['nombre_clusters'] for langue in langues]
nombre_clusters_4 = [donnees_4[langue]['nombre_clusters'] for langue in langues]
nombre_clusters_5 = [donnees_5[langue]['nombre_clusters']*2 for langue in langues]

# Création des sous-graphiques par langue
fig, axs = plt.subplots(len(langues), figsize=(10, 6*len(langues)), sharex=True)

# Ajout des données à chaque sous-graphique
for i, langue in enumerate(langues):
    axs[i].bar(['B', 'T', '4', '5'], [nombre_clusters_B[i], nombre_clusters_T[i], nombre_clusters_4[i], nombre_clusters_5[i]], color=['skyblue', 'lightgreen', 'coral', 'gold'])
    axs[i].set_ylabel('Nombre de clusters')
    axs[i].set_title(f'Nombre de clusters pour la langue {langue}')

plt.xlabel('Type de fichier')
plt.tight_layout()

# Vérifie si le dossier 'graphiquesetape6et7langue' existe, sinon le crée
if not os.path.exists('graphiquesetape6et7langue'):
    os.makedirs('graphiquesetape6et7langue')

# Enregistre chaque figure dans le dossier 'graphiquesetape6et7langue'
for index, langue in enumerate(langues):
    plt.savefig(f"graphiquesetape6et7langue/cluster_{langue}.png")

plt.show()
