# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:21:25 2024

@author: ilian
"""

import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split
import string

def get_nom(chemin):
    fichier1 = chemin.split('/')
    fichier2 = fichier1[-1]
    return fichier2 



input_files = []

input_files.append("edgar_allan_poe.txt")
input_files.append("robert_frost.txt")
# print(input_files)


input_texts = []
labels = []

for elm in enumerate(input_files):
    f = elm[1]
    label = elm[0]
    for tex in open(f, "r", encoding = "utf-8"):
        # on donne f comme argument parce que c'est la variable qui contient le texte
        tex = tex.lower()
        # print(tex)
        tex = tex.rstrip()
        # print("*******", tex)
        for char in tex: 
            if char in string.punctuation: 
                tex = tex.replace(char, "")
        line = tex
        # print(tex)
        input_texts.append(line)
        labels.append(label)
        
# print(labels, input_texts)

# Parte 1.3

train_text , test_text , Ytrain , Ytest = train_test_split(input_texts, labels)

# print(len(Ytrain), len(Ytest)) #ceci montre la quantité de phrases d'entrainement et d'évaluation qu'il y a au total
# print(train_text[:5]) #ceci montre les 5 premières lignes de texte d'entrainement
# print(Ytrain[:5]) #ceci montre les 5 premières étiquettes des textes d'entrainement

# Parte 1.4

idx = 1
word2idx = {'<unk>': 0}
for line in train_text:
    tokens = line.split(' ')
    # print(tokens)
    for word in tokens:
        if word not in word2idx:
            word2idx[word] = idx
            idx +=1
            
# print(word2idx)

# Parte 1.5

train_text_int = []
test_text_int = []

for elm in train_text:
    spl_line = elm.split(' ')
    lis_temp = []
    for word in spl_line:
        lis_temp.append(word2idx[word])
    train_text_int.append(lis_temp)
    
# print(train_text_int)

# Repetir todo para las données de evaluación

idx = 1
word3idx = {'<unk>': 0}
for line in test_text:
    tokens = line.split(' ')
    # print(tokens)
    for word in tokens:
        if word not in word3idx:
            word3idx[word] = idx
            idx +=1
            

for elm in test_text:
    spl_line = elm.split(' ')
    lis_temp = []
    for word in spl_line:
        lis_temp.append(word3idx[word])
    test_text_int.append(lis_temp)
    
# print(test_text_int)


print(train_text_int[50], train_text_int[100], train_text_int[150], train_text_int[200])
print("***************************************************************************")
print(test_text_int[50], test_text_int[100], test_text_int[150], test_text_int[200])

# Je n'ai pas compris pourquoi comparer ces 2 informations puisque de toutes façons ce sont 2 parties différentes du texte, nous savons déjà qu'elles seront différentes

