#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:46:25 2024

@author: ceres
"""
from TD2_CKP import *

modele="sm"
nlp = spacy.load(f"fr_core_news_{modele}")
path_corpus="../ressources/ressources_TD1_Entite-nommee/Texte/*/*/*.txt"

for path in glob.glob(path_corpus):
    print(path)
    texte=lire_fichier(path)
    tok_split=tokenisation_split(texte)
    doc = nlp(texte)
    
    tok_spacy=tokenisation_spacy(doc)
    lem=lemmatisation(doc)
    postag= POStag(doc)
    stocker_json("%s_token_split.json"%path, tok_split)
    stocker_json("%s_token_%s.json"%(path,modele), tok_spacy)
    stocker_json("%s_lemma_%s.json"%(path,modele), lem)
    stocker_json("%s_PosTag_%s.json"%(path,modele), postag)