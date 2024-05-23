#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:16:20 2022

@author: antonomaz
"""


import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import DistanceMetric 
from sklearn.feature_extraction.text import CountVectorizer
import sklearn
import json
import glob
import re
from collections import OrderedDict
import matplotlib.pyplot as plt


def sauvegarder_json(dico, nom_liste, langue):
    nom_fichier = f"{nom_liste}_{langue}.json"
    with open(nom_fichier, "w", encoding="utf-8") as json_file:
        json.dump(dico, json_file, ensure_ascii=False, indent=2)
    print(f"Vocabulaire sauvegardé dans {nom_fichier} avec succès !")
        
def n_gramme(listee,nb=int(input("donner in chiffre"))):
    nbr_gram=[] 
    for elt in listee:
        if len(elt) > 1:
            nbr_gram.append(elt[:nb])
    return nbr_gram

def nomfichier(chemin):
    nomfich= chemin.split("/")[-1]
    nomfich= nomfich.split(".")
    nomfich= ("_").join([nomfich[0],nomfich[1]])
    return nomfich
    
def lire_json(chemin_fichier_json):
    with open(chemin_fichier_json, "r", encoding="utf-8") as r:
        fich_json = json.load(r)
    return fich_json
 

chemin_entree = ["vocab_manuel_français.json","vocab_auto_français.json","vocab_manuel_anglais.json","vocab_auto_anglais.json"]   


def cluster(chemin):
#for subcorpus in glob.glob(path_copora):
#    print("SUBCORPUS***",subcorpus)
 #   liste_nom_fichier =[]
    for path in glob.glob(chemin):
    #        print("PATH*****",path)
            liste_nom_fichier=[]
            nom_fichier = nomfichier(path)
    #        print(nom_fichier)
            liste= lire_json(path)
            liste_n_gramme= n_gramme(liste)
            print("les n_grammes ont été récupéré")
            
    #### FREQUENCE ########
            
            dic_mots={}
            i=0
        
            
            for mot in liste_n_gramme: 
                
                if mot not in dic_mots:
                    dic_mots[mot] = 1
                else:
                    dic_mots[mot] += 1
            
            i += 1
    
            new_d = OrderedDict(sorted(dic_mots.items(), key=lambda t: t[0]))
            
            freq=len(dic_mots.keys())
            
    
            Set_00 = set(liste_n_gramme)
            Liste_00 = list(Set_00)
            dic_output = {}
            liste_words=[]
            matrice=[]
            
            for l in Liste_00:
                    
                if len(l)!=1:
                    liste_words.append(l)
            
    
            try:
                words = np.asarray(liste_words)
                for w in words:
                    liste_vecteur=[]
                
                        
                    for w2 in words:
                    
                            V = CountVectorizer(ngram_range=(2,3), analyzer='char')
                            X = V.fit_transform([w,w2]).toarray()
                            distance_tab1=sklearn.metrics.pairwise.cosine_distances(X)            
                            liste_vecteur.append(distance_tab1[0][1])
                        
                    matrice.append(liste_vecteur)
                matrice_def=-1*np.array(matrice)
               
                      
                affprop = AffinityPropagation(affinity="precomputed", damping= 0.6, random_state = None) 
        
                affprop.fit(matrice_def)
                for cluster_id in np.unique(affprop.labels_):
                    exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
                    cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
                    cluster_str = ", ".join(cluster)
                    cluster_list = cluster_str.split(", ")
                                
                    Id = "ID "+str(i)
                    for cle, dic in new_d.items(): 
                        if cle == exemplar:
                            dic_output[Id] ={}
                            dic_output[Id]["Centroïde"] = exemplar
                            dic_output[Id]["Freq. centroide"] = dic
                            dic_output[Id]["Termes"] = cluster_list
                    
                    i=i+1
                return dic_output
    
                
    
            except :        
               # print("**********Non OK***********", path)
    
        
                liste_nom_fichier.append(path)
                
                
               # continue 
    

#bi_gramme_vocab_manuel_fr= sauvegarder_json(cluster(chemin_entree[0]), "bi_gramme_vocab_manuel", "fr")
#tri_gramme_vocab_manuel_fr= sauvegarder_json(cluster(chemin_entree[0]),"tri_gramme_vocab_manuel","fr")   
#quatre_gramme= sauvegarder_json(cluster(chemin_entree[2]), "quatre_gramme_vocab_manuel", "ang")
#cinq_gramme= sauvegarder_json(cluster(chemin_entree[2]),"cinq_gramme_vocab_manuel","ang")

#REPRESENTATION GRAPHIQUE DES CLUSTERS
 #liste où on stockera les donnees pour le graphique
cluster_len= []
centroid= []
 
entree_json= "les_n_grammes_json/cinq_gramme_vocab_manuel_ang.json"
 
for path2 in glob.glob(entree_json):
    fichier= lire_json(path2)
    
    # calculer la longueure des clusters et les ajouter dans liste
    for cluster2 in fichier:
        cluster_length= len(fichier[cluster2]["Termes"])
        cluster_len.append(cluster_length)
        centroid.append(fichier[cluster2]["Centroïde"])
        
#realisation des graphiques
plt.figure(figsize=(20,14)) #taille du graphique
for i , (length, centroids) in enumerate(zip(cluster_len,centroid)):
    plt.scatter(i,length,s= length*15, alpha=0.7, label=f"Cluster{i}") #personnaliser les cercles du graphqiues
    plt.annotate(centroids,(i,length),textcoords= "offset points", xytext=(0,6), ha="center",fontsize="8")# personaliser le titre des cercles
    
plt.xlabel("cluster index")
plt.ylabel("longueur des clusters")
plt.title("taille des clusters pour cinq-grammes ang")
plt.grid(True)
#plt.savefig("graphique cinq-gramme ang")
#plt.show()
    