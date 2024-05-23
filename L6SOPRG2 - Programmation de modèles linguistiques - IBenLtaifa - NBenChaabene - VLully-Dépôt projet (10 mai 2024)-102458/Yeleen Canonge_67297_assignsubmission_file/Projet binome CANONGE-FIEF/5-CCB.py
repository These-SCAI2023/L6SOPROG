#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:33:47 2024

@author: Yeleen
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
import json
import glob
import os

def lire_fichier(chemin):
    with open(chemin) as json_data:
        texte = json.load(json_data)
    return texte

chemin_entree = "donneesjsonB"  # Chemin d'entrée où se trouvent les fichiers JSON à traiter

# Fonction pour effectuer le clustering hiérarchique par langue
def cluster_hierarchique_par_langue(data, langue):
    if not data:
        return [], [], [], []  # Retourner des listes vides si aucune donnée pour la langue donnée
    
    # Conversion des données en vecteurs (pas nécessaire dans ce cas)
    vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer='char')
    X = vectorizer.fit_transform(data)
    
    # Calcul des distances entre les échantillons
    distances = cosine_distances(X)
    
    # Clustering hiérarchique
    clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete', distance_threshold=0.7)
    clustering.fit(distances)
    
    # Extraction des clusters
    clusters_centroides = []
    clusters_mots = []
    clusters_nombre_mots = []
    clusters_frequence_totale = []
    
    for cluster_id in np.unique(clustering.labels_):
        cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
        cluster = [data[i] for i in cluster_indices]
        exemplar = cluster[0]
        frequence_totale = sum([data.count(mot) for mot in cluster])
        clusters_centroides.append(exemplar)
        clusters_mots.append(cluster)
        clusters_nombre_mots.append(len(cluster))
        clusters_frequence_totale.append(frequence_totale)
    
    return clusters_centroides, clusters_mots, clusters_nombre_mots, clusters_frequence_totale

# Dictionnaire pour stocker les données de chaque langue
donnees_par_langue = {}

# Boucle sur les langues
for langue in ["en", "es", "fr"]:
    data_langue = []
    # Boucle sur les fichiers JSON
    json_files = glob.glob(os.path.join(chemin_entree, langue, "*", "*.json"))
    for json_file in json_files:
        data_langue += lire_fichier(json_file)
    
    # Effectuer le clustering hiérarchique par langue
    clusters_centroides, clusters_mots, clusters_nombre_mots, clusters_frequence_totale = cluster_hierarchique_par_langue(data_langue, langue)
    
    # Stocker les données dans le dictionnaire
    donnees_par_langue[langue] = {
        'nombre_clusters': len(clusters_centroides),
        'centroids': clusters_centroides,
        'bigrammes': clusters_mots,
        'bigramme_counts': clusters_nombre_mots,
        'total_bigramme_frequencies': clusters_frequence_totale
    }

# Chemin de sauvegarde du fichier JSON
chemin_sortie_json = "clusters_hierarchiqueB.json"

# Écriture du dictionnaire dans un fichier
with open(chemin_sortie_json, 'w') as json_file:
    json.dump(donnees_par_langue, json_file, indent=4)
