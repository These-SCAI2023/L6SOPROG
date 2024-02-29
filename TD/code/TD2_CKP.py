#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:58:59 2024

@author: ceres
"""
import json
import glob
import spacy

def lire_fichier(chemin):
    with open(chemin, encoding="utf-8") as f:
        chaine = f.read()
    return chaine

def tokenisation(txt_analyse):
    liste_token=[]
    for token in txt_analyse:
        liste_token.append(token.text)
    return liste_token

def lemmatisation(txt_anly):
    liste_lem=[]
    for token in txt_anly:
        liste_lem.append(token.lemma_) 
    return liste_lem

def POStag(txt_anl):
    liste_POS=[]
    for token in doc:
        liste_POS.append([token.text, token.lemma_, token.pos_, token.tag_, token.dep_,token.shape_, token.is_alpha, token.is_stop])
    return liste_POS

def stocker_json(chemin,contenu):
    with open(chemin, "w", encoding="utf-8") as w:
        w.write(json.dumps(contenu, indent =2,ensure_ascii=False))
    return

modele="sm"
nlp = spacy.load(f"fr_core_news_{modele}")
path_corpus="../ressources_TD1_Entite-nommee/Texte/*/*/*.txt"

for path in glob.glob(path_corpus):
    print(path)
    texte=lire_fichier(path)
    doc = nlp(texte)
    tok=tokenisation(doc)
    lem=lemmatisation(doc)
    postag= POStag(doc)
    stocker_json("%s_token_%s.json"%(path,modele), tok)
    stocker_json("%s_lemma_%s.json"%(path,modele), lem)
    stocker_json("%s_PosTag_%s.json"%(path,modele), postag)

