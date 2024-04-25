# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:14:44 2024

@author: -
"""


import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split
import glob

################# EXERCICE 1

input_texts = []
labels = []
input_files = ["robert_frost.txt", "edgar_allan_poe.txt"]

# Créez une boucle en utilisant enumerate() pour itérer sur input_files
for label, f in enumerate(input_files):
    # print(label)
    # print(f)
    with open(f, "r") as f:
        for ligne in f:
            ligne = ligne.lower()
            ligne = ligne.rstrip()
            line = ligne.translate(str.maketrans("", "", string.punctuation))
            # print(line)
            input_texts.append(line)
            labels.append(label)
# print(input_texts)
# print(labels)
    

train_text, test_text, Ytrain, Ytest = train_test_split(input_texts, labels, test_size = 0.3)
len(Ytrain), len(Ytest)
train_text[:5]
Ytrain[:5]


#on initialise le dictionnaire word2idx avec un indice pour les mots inconnus + initialisation idx
word2idx = {'<unk>': 0}
idx = 1

#boucle pour construire le vocabulaire
for line in train_text:
    #on séapre la ligne en mots
    tokens = line.split()
    #on parcourt les mots dans la ligne
    for token in tokens:
        #si le mot n'est pas déjà dans word2idx
        if token not in word2idx:
            #on l'ajoute au dictionnaire avec son indice 
            word2idx[token] = idx
            #incrémentation de l'indice
            idx += 1


# print("Contenu word2idx :", word2idx)
# print("Taille vocabulaire :", len(word2idx))
    
    
#on converti les données d'entraînement
train_text_int = []
for line in train_text:
    #liste pour stocker les représentations numériques des mots dans la ligne
    temp_line = []
    #on sépare la ligne en mots
    tokens = line.split()
    #on parcourt les mots de la ligne
    for token in tokens:
        #pour trouver la représentation numérique du mot dans word2idx
        if token in word2idx:
            temp_line.append(word2idx[token])
        else:
            #si le mot n'est pas dans le vocabulaire, utiliser l'indice du mot inconnu
            temp_line.append(word2idx['<unk>'])
    #on ajoute la liste temporaire à train_text_int
    train_text_int.append(temp_line)

#on converti les données d'évaluation 
test_text_int = []
for line in test_text:
    #liste temporaire pour stocker les représentations numériques des mots dans la ligne
    temp_line = []
    #on sépare la ligne en mots
    tokens = line.split()
    #on parcourt les mots dans la ligne
    for token in tokens:
        #on trouve la représentation numérique du mot dans word2idx
        if token in word2idx:
            temp_line.append(word2idx[token])
        else:
            #si le mot n'est pas dans le vocabulaire, utiliser l'indice du mot inconnu
            temp_line.append(word2idx['<unk>'])
    test_text_int.append(temp_line)

#on compare les différentes lignes 
for i in range(50, 200, 50):
    print("Ligne", i, "de train_text_int :", train_text_int[i])
    print("Ligne", i, "de train_text :", train_text[i])


        

############# EXERICE 2
    
V = len(word2idx)

#création des matrices A0 et A1 initialisées à 1
A0 = np.ones((V, V))
A1 = np.ones((V, V))

#création des listes pi0 et pi1 initialisées à 1
pi0 = np.ones(V)
pi1 = np.ones(V)

def compute_counts(text_as_int, A, pi):
    #on aprcourt chaque séquence de mots dans text_as_int
    for tokens in text_as_int:
        last_idx = None  #on initialise l'indice du dernier mot à None pour la première itération
        for idx in tokens:
            if last_idx is None:
                #le premier mot de la phrase, on incrémente le compteur correspondant dans pi
                pi[idx] += 1
            else:
                #pour compter les transitions pour la transition du dernier mot au mot actuel
                A[last_idx, idx] += 1
            #mise à jour de l'indice du dernier mot
            last_idx = idx

compute_counts([t for t, y in zip(train_text_int, Ytrain) if y == 0], A0, pi0)
