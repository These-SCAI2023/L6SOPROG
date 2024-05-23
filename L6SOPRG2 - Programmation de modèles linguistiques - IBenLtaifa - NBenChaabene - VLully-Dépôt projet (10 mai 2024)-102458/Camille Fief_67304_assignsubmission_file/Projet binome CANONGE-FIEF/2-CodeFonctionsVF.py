#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:32:09 2024

@author: Yeleen
"""

import glob
import spacy
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import json

# Mapping des langues aux modèles SpaCy correspondants
modeles_langue = {
    'en': 'en_core_web_sm',
    'fr': 'fr_core_news_sm',
    'es': 'es_core_news_sm'
}

# Fonction pour lire les fichiers texte
def lire_fichier(dossier_corpus):
    fichiers = []
    # Chemin vers les fichiers texte dans le dossier corpusM
    chemin_fichiers_texte = f"{dossier_corpus}/*/*/*.html"

    # Utiliser glob pour récupérer la liste des chemins des fichiers texte
    for chemin_fichier in glob.glob(chemin_fichiers_texte):
        # Lire le contenu du fichier
        with open(chemin_fichier, 'r', encoding='utf-8') as fichier:
            texte = fichier.read()
        
        # Extraire la langue à partir du nom du fichier
        langue = chemin_fichier.split("/")[-3]
        fichiers.append((texte, langue))

    return fichiers


# Fonction pour charger les modèles SpaCy
def charger_modele_spacy(langue):
    if langue not in modeles_langue:
        print(f"Langue non supportée: {langue}")
        return None
    return spacy.load(modeles_langue[langue])

# Fonction qui tokenise les textes
def tokenizer_texte(texte, langue= None):
    nltk.download('punkt')
    # Tokenize le texte en utilisant le tokenizer punkt de nltk
    return word_tokenize(texte)

# Fonction qui lemmatise le texte
def lemmatiser_texte(tokens, langue= None):
    nlp = charger_modele_spacy(langue)
    if nlp is None:
        return []
    # Lemmatize chaque token dans la liste
    return [token.lemma_ for token in nlp(" ".join(tokens))]

# Fonction qui effectue la REN
def reconnaitre_entites_nommees(texte, langue= None):
    nlp = charger_modele_spacy(langue)
    if nlp is None:
        return []
    # Effectuer la REN sur le texte
    doc = nlp(texte)
    # Récupérer les entités nommées détectées dans le texte
    return [(ent.text, ent.label_) for ent in doc.ents]

# Fonction qui permet de supprimer les mots vides et les noms propres des textes
def filtrer_mots(texte, langue= None):

    # Charger le modèle SpaCy correspondant à la langue spécifiée
    nlp = charger_modele_spacy(langue)
    if nlp is None:
        return [], []

    # Effectuer le traitement du texte avec SpaCy
    doc = nlp(texte)

    # Exclure les mots vides et conserver les noms propres
    tokens_filtres = [token.text for token in doc if not token.is_stop]
    noms_propres = [token.text for token in doc if token.pos_ == 'PROPN']

    return tokens_filtres, noms_propres


def recup_info(fichiers):
    info = {}
    for texte, langue in fichiers:
        # Tokenisation
        tokens = word_tokenize(texte)

        # Nombre de tokens par texte pour chaque langue
        info.setdefault(langue, {}).setdefault('nb_tokens_par_texte', []).append(len(tokens))

        # Nombre de token type (vocabulaire) par texte pour chaque langue
        nb_token_type = len(set(tokens))
        info.setdefault(langue, {}).setdefault('nb_token_type_par_texte', []).append(nb_token_type)

        # Proportion de lemmes par texte pour chaque langue
        nlp = charger_modele_spacy(langue)
        if nlp is None:
            continue
        lemmes = [token.lemma_ for token in nlp(" ".join(tokens))]
        proportion_lemmes = len(lemmes) / len(tokens)
        info.setdefault(langue, {}).setdefault('proportion_lemmes_par_texte', []).append(proportion_lemmes)

        # Nombre de lemmes total par langue
        nb_lemmes_total = len(lemmes)
        info.setdefault(langue, {}).setdefault('nb_lemmes_total', 0)
        info[langue]['nb_lemmes_total'] += nb_lemmes_total

        # Proportion de noms propres pour chaque langue
        noms_propres = [token.text for token in nlp(" ".join(tokens)) if token.pos_ == 'PROPN']
        proportion_noms_propres = len(noms_propres) / len(tokens)
        info.setdefault(langue, {}).setdefault('proportion_noms_propres_par_texte', []).append(proportion_noms_propres)

    # Calcul du nombre de tokens global par langue
    for langue, values in info.items():
        values['nb_tokens_global'] = sum(values['nb_tokens_par_texte'])

    # Sauvegarde des informations dans un fichier JSON
    sauvegarder_info(info)

    return info

def sauvegarder_info(info):
    with open('informationsV1.2C.json', 'w') as f:
        json.dump(info, f, indent=4)


# Exploitation
dossier_corpus = "corpusM"
fichiers = lire_fichier(dossier_corpus)
# Prétraitement des fichiers
for texte, langue in fichiers:
    # Tokenisation et lemmatisation du texte
    tokens = tokenizer_texte(texte, langue=langue)
    tokens_lemmatises = lemmatiser_texte(tokens, langue=langue)
    
    # REN du texte
    entites_nommees = reconnaitre_entites_nommees(texte, langue=langue)
    
    # Filtrage des mots vides et des noms propres
    tokens_filtres, noms_propres = filtrer_mots(texte, langue=langue)

# Exploitation
infos = recup_info(fichiers)
sauvegarder_info(infos)

    #Affichage des informations
    # print("Langue du fichier :", langue)
    # print("Tokens lemmatisés :", tokens_lemmatises)
    # print("Entités nommées :", entites_nommees)
    # print("Tokens filtrés :", tokens_filtres)
    # print("Noms propres :", noms_propres)
    # print("------------------------------")
