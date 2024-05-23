#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:52:37 2024

@author: Yeleen
"""

import glob
import json
import os
import re


# Fonction pour nettoyer les données textuelles
def nettoyer_texte(texte, langue=None):
    # Supprimer la ponctuation et les chiffres
    texte = re.sub(r'[^a-zA-Z]', '', texte)
    # Conversion en minuscules
    texte = texte.lower()
    # Génération de bi-grammes de lettres consécutives
    bi_grams = [texte[i:i+2] for i in range(len(texte)-1)]
    return bi_grams


def process_files(folder_path, max_files_per_language=20):
    data = {}
    files_per_language = {'fr': 0, 'en': 0, 'es': 0}  # Compteur pour les fichiers traités par langue
    for file_path in glob.glob(os.path.join(folder_path, "corpusM", "**", "**", "*.html"), recursive=True):
        # Extraire la langue du chemin du fichier
        lang = file_path.split(os.path.sep)[-3]
        if lang not in files_per_language:
            continue  # Ignorer les langues autres que fr, en, es
        
        # Vérifier si le nombre maximum de fichiers par langue a été atteint
        if files_per_language[lang] >= max_files_per_language:
            continue
        
        with open(file_path, 'r', encoding='utf-8') as file:
            file_data = file.read().strip()
            
            # Nettoyage des données textuelles
            tokens = nettoyer_texte(file_data, lang)
            
            # Utilisation du chemin relatif comme clé
            relative_path = os.path.relpath(file_path, os.path.join(folder_path, "corpusM"))
            data[relative_path] = set(tokens)
            
            # Incrémenter le compteur de fichiers traités pour cette langue
            files_per_language[lang] += 1
            
    return data

# Si ce script est exécuté en tant que programme principal
if __name__ == "__main__":
    # Récupère le chemin du répertoire actuel
    folder_path = '.'
    # Crée un chemin pour le répertoire de sortie "donneesjson" dans le répertoire actuel
    output_folder = os.path.join(folder_path, "donneesjsonB")
    # Crée le répertoire de sortie s'il n'existe pas déjà
    os.makedirs(output_folder, exist_ok=True)
    
    # Traite les fichiers dans le répertoire actuel et ses sous-répertoires
    processed_data = process_files(folder_path)
    
    # Parcourt les données traitées
    for filename, file_data in processed_data.items():
        # Crée le chemin de sortie pour chaque fichier JSON en conservant la structure des dossiers
        output_path = os.path.join(output_folder, filename.replace('.html', '.json'))
        # Crée les dossiers nécessaires pour le fichier de sortie
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Ouvre chaque fichier JSON en mode écriture, spécifiant l'encodage utf-8
        with open(output_path, 'w', encoding='utf-8') as outfile:
            # Convertit les données du fichier en une liste de tokens et les écrit dans le fichier JSON
            json.dump(list(file_data), outfile, ensure_ascii=False, indent=4)
