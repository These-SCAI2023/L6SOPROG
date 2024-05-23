#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:24:17 2024

@author: Yeleen
"""
import json
import matplotlib.pyplot as plt
import os

# Fonction pour créer un graphique à partir d'un fichier JSON
def creer_graphique(fichier_json):
    # Chargement des données depuis le fichier JSON
    with open(fichier_json) as json_file:
        donnees = json.load(json_file)

    # Récupération du nombre de clusters par langue
    langues = list(donnees.keys())
    nombre_clusters = [donnees[langue]['nombre_clusters'] for langue in langues]

    # Si le fichier se termine par "5", multipliez le nombre de clusters par 2
    if fichier_json.endswith("5.json"):
        nombre_clusters = [n * 2 for n in nombre_clusters]

    # Création du graphique à barres
    plt.figure(figsize=(10, 6))
    bars = plt.bar(langues, nombre_clusters, color='skyblue')
    plt.xlabel('Langue')
    plt.ylabel('Nombre de clusters')
    plt.title(f'Nombre de clusters par langue - {fichier_json}')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajout des nombres exacts au-dessus des barres
    for bar, nombre in zip(bars, nombre_clusters):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(nombre), ha='center', va='bottom', fontsize=9)

    # Vérifie si le dossier 'graphiquesetape6et7ngramme' existe, sinon le crée
    if not os.path.exists('graphiquesetape6et7ngramme'):
        os.makedirs('graphiquesetape6et7ngramme')

    # Enregistre le graphique dans le dossier 'graphiquesetape6et7ngramme'
    plt.savefig(f"graphiquesetape6et7ngramme/{fichier_json.split('.')[0]}.png")
    plt.close()

# Liste des fichiers à traiter
fichiers_a_traiter = ["clusters_hierarchiqueB.json", "clusters_hierarchiqueT.json", "clusters_hierarchique4.json", "clusters_hierarchique5.json"]

# Création du graphique pour chaque fichier
for fichier in fichiers_a_traiter:
    creer_graphique(fichier)
