#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:20:13 2024

@author: Yeleen
"""

import nltk
import spacy

def download_nltk_resources():
    # Télécharger les ressources nécessaires pour nltk
    nltk.download('punkt')

def download_spacy_models():
    # Télécharger les modèles SpaCy pour les trois langues
    languages = ['en_core_web_sm', 'fr_core_news_sm', 'es_core_news_sm']
    for lang in languages:
        spacy.cli.download(lang)

# Télécharger les ressources nécessaires pour nltk
download_nltk_resources()

# Télécharger les modèles SpaCy pour les trois langues
download_spacy_models()
