# -*- coding: utf-8 -*-
"""
Created on Wed May  8 20:11:19 2024

@author: user
"""

import os
import json
import matplotlib.pyplot as plt

# Vérifier si le dossier de sauvegarde existe, sinon le créer
save_dir = 'graphiques étape 5'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Chargement des données depuis le fichier JSON
with open('informations.json', 'r') as f:
    data = json.load(f)

# Fonction pour créer les nuages de points pour chaque langue
def plot_metrics_scatter(langue):
    # Récupération des données pour la langue donnée
    nb_tokens = data[langue]['nb_tokens_par_texte']
    nb_token_type = data[langue]['nb_token_type_par_texte']
    proportion_lemmes = data[langue]['proportion_lemmes_par_texte']
    proportion_noms_propres = data[langue]['proportion_noms_propres_par_texte']

    # Création du nuage de points pour le nombre de tokens par texte
    plt.figure(figsize=(10, 8))
    plt.scatter(range(1, len(nb_tokens) + 1), nb_tokens, color='blue')
    plt.title(f'Nombre de tokens par texte ({langue})')
    plt.xlabel('Texte')
    plt.ylabel('Nombre de tokens')
    plt.savefig(os.path.join(save_dir, f'nb_tokens_{langue}.png'))
    plt.close()

    # Création du nuage de points pour le nombre de token types par texte
    plt.figure(figsize=(10, 8))
    plt.scatter(range(1, len(nb_token_type) + 1), nb_token_type, color='blue')
    plt.title(f'Nombre de token types par texte ({langue})')
    plt.xlabel('Texte')
    plt.ylabel('Nombre de token types')
    plt.savefig(os.path.join(save_dir, f'nb_token_type_{langue}.png'))
    plt.close()

    # Création du nuage de points pour la proportion de lemmes par texte
    plt.figure(figsize=(10, 8))
    plt.scatter(range(1, len(proportion_lemmes) + 1), proportion_lemmes, color='blue')
    plt.title(f'Proportion de lemmes par texte ({langue})')
    plt.xlabel('Texte')
    plt.ylabel('Proportion de lemmes')
    plt.savefig(os.path.join(save_dir, f'proportion_lemmes_{langue}.png'))
    plt.close()

    # Création du nuage de points pour la proportion de noms propres par texte
    plt.figure(figsize=(10, 8))
    plt.scatter(range(1, len(proportion_noms_propres) + 1), proportion_noms_propres, color='blue')
    plt.title(f'Proportion de noms propres par texte ({langue})')
    plt.xlabel('Texte')
    plt.ylabel('Proportion de noms propres')
    plt.savefig(os.path.join(save_dir, f'proportion_noms_propres_{langue}.png'))
    plt.close()

# Création des nuages de points pour chaque langue
plot_metrics_scatter('fr')  # Français
plot_metrics_scatter('en')  # Anglais
plot_metrics_scatter('es')  # Espagnol
