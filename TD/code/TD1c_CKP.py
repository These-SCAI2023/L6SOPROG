#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:51:13 2024

@author: ceres
"""

#Langue proche

import glob
from TD1_CKP import *
import pandas as pd
import numpy 
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

def stocker_png(chemin,contenu):
    fig, ax = plt.subplots(figsize=(10,10))
    hm = sns.heatmap(contenu)

    plt.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.clf()
    

path_mod="./model/"

dic_proche={}

for pth_md in glob.glob(f"{path_mod}/*10.json"):
    # print(pth_md)
    if "test" in pth_md:
        model_test=lire_json(pth_md)
    else:
        model_appr=lire_json(pth_md)

liste_k=[]
  
liste_taux_taux=[] 
for k,v in model_test.items():
    liste_k.append(k)
    liste_taux=[]
    liste_key=[]
    for key,val in model_appr.items():
        liste_key.append(key)
        
        ## En utilisant les intersections
        v=set(v)
        val=set(val)
        # compare_mod=set(v).intersection(set(val))
        compare_mod=v.intersection(val)
        compare_mod2=val.intersection(v)
        taux=k,key,len(compare_mod)*10
        
  
        liste_taux.append(taux[-1])
    liste_taux_taux.append(liste_taux)

##Je fabrique la matrice carrée qui comprends le nom de chaque document, le nom de chaque langue et le score 
# ar = numpy.array(liste_taux_taux)
    
# ## Je fabrique le Tableau

# df = pandas.DataFrame(ar, index = liste_key, columns = liste_k)
# stocker_png("./corpus_multi/langues_proches2.png",df)
# ## Je produis la Heatmap

# fig, ax = plt.subplots(figsize=(10,10))
# hm = sns.heatmap(df)

# plt.savefig("./corpus_multi/langues_proches.png", dpi=300, bbox_inches="tight")
# plt.clf()

  
