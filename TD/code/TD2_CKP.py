#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:58:59 2024

@author: ceres
"""
import json
import glob
import spacy
import matplotlib.pyplot as pyplot
from functools import reduce

def lire_fichier(chemin):
    with open(chemin, encoding="utf-8") as f:
        chaine = f.read()
    return chaine

def lire_fichier_json (chemin):
    with open(chemin) as json_data: 
        texte =json.load(json_data)
    return texte


def tokenisation_split(txt):
    tok=txt.split(" ")
    return tok

def tokenisation_spacy(txt_analyse):
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
    for token in txt_anl:
        liste_POS.append([token.text, token.lemma_, token.pos_, token.tag_, token.dep_,token.shape_, token.is_alpha, token.is_stop])
    return liste_POS

def stocker_json(chemin,contenu):
    with open(chemin, "w", encoding="utf-8") as w:
        w.write(json.dumps(contenu, indent =2,ensure_ascii=False))
    return


def texte_to_dict(texte):
    texte_dict = {}
    
    for token in texte:
        if token in texte_dict:
            texte_dict[token] += 1
        else:
            texte_dict[token] = 1

    return texte_dict


def dict_to_list(texte_dict):
    texte_list=[]

    for mot in texte_dict.keys():
        texte_list.append([texte_dict[mot], mot])    

    texte_list.sort(reverse=True)
    return texte_list


def afficher_n(texte_list, n):
    
    cumul = 0
    print("rang\tmot\tfrequence\tfrequence(Zipf)")    
    print("-"*50)
    for _ in range(n):
        cumul += texte_list[_][0]
        print("{}\t{}\t{}\t\t{:.0f}".format(_+1, texte_list[_][1], texte_list[_][0], texte_list[0][0]/(_+1)))
    
    total = reduce(lambda x, y: x+y, [_[0] for _ in texte_list])
    prop = cumul/total*100
    
    print("-"*50)
    print("Ces {} mots représentent le {:0.2f}% du corpus".format(n, prop))

    
def plot_zipf(texte_list1,texte_list2,ch,name_output, log=False):
    pyplot.rcParams['figure.figsize'] = [15, 10]

    y1 = [_[0] for _ in texte_list1]
    y2 = [_[0] for _ in texte_list2]

   # y_ = []
  #  for _ in range(len(texte_list)):
   #     y_.append(int(texte_list[0][0]/(_+1)))    

    pyplot.plot(y1, "-", label="spaCy token")
    pyplot.plot(y2, "--", label="split token")
    #pyplot.plot(y_, "--", label="Approximation (Zipf)") 
    
    if log:
        pyplot.yscale("log")
        pyplot.xscale("log")     
      
    pyplot.legend()
    pyplot.title("Loi de Zipf (Brown Corpus)")
    pyplot.xlabel("Rang")
    pyplot.ylabel("Fréquence")
    pyplot.savefig(f"{ch}/{name_output}", dpi=300)##Pour stocker l'image sur sa machine
    pyplot.show()    


