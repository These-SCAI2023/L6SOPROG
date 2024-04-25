#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Exercice 1


# In[7]:


import numpy as np
import matplotlib . pyplot as plt
import string
from sklearn.model_selection import train_test_split
import os
import string


# In[8]:


#lecture des fichiers et enregistrement des noms dans liste (étape 1.2.1)

input_files=[file for file in os.listdir() if file.endswith('.txt')]

for file in input_files:
    with open(file, 'r') as f:
        first_lines = [next(f) for _ in range(5)]
        print(f"Contenu du fichier {file}:")
        for line in first_lines:
            print(line.strip())
        print("\n")
        
print(input_files) #on vérifie que les noms ont été stockés correctement


# In[9]:


#lecture des corpus (étape 1.2.2)
input_texts=[]
labels=[]

#boucle pour parcourir les fichiers et leurs index
for idx, file in enumerate(input_files):
    label=idx #on utilise l'index comme étiquette
    with open(file, 'r') as f:
        #on parcourt les lignes du fichier
        for line in f:
            #prétraitement de la ligne
            line=line.lower().rstrip()
            line=line.translate(str.maketrans('', '', string.punctuation))
            #ajout de la ligne prétraitée et l'étiquette aux listes correspondantes
            input_texts.append(line)
            labels.append(label)


# In[10]:


#division de corpus (étape 1.3)

train_text, test_text, Ytrain, Ytest = train_test_split(input_texts, labels, test_size=0.3, random_state=42)

#on vérifie la taille des ensembles entraînement et test
print("Taille de Ytrain:", len(Ytrain))
print("Taille de Ytest:", len(Ytest))

#on affiche les  premiers textes et leurs étiquettes d'entraînelent
print("Cinq premiers textes d'entraînement:")
print(train_text[:5])
print("Cinq premières étiquettes d'entraînement:")
print(Ytrain[:5])

#len(Ytrain) et len(Ytest) montre la taille de chacun des enesmbles, on peut alors vérifier que la règle des 70/30 a bien été respectée
#les 2 autres commandes affichent les 5 premiers textes de l'ensemble d'entraînement et des étiquettes qui y sont associées respectivement


# In[11]:


#mapping (étape 1.4)

idx=1
word2idx={'<unk>':0}

#on parcourt le texte d'entraînement pour construire le vocabulaire
for line in train_text:
    tokens=line.split()#on découpe la ligne en mots
    for token in tokens:
        if token not in word2idx:
            word2idx[token]=idx
            idx+=1

#on affiche le contenu du dico word2idx
print("Cotenu du dictionnaire word2idx:")
print(word2idx)

#taille du vocabulaire
vocab_size=len(word2idx)
print("Taille de notre vocabulaire:", vocab_size)


# In[14]:


#vérification des lignes vides dans train_text
for i, line in enumerate(train_text):
    if len(line) == 0:
        print(f"Ligne {i} de train_text est vide.")

#vérification des lignes vides dans test_text
for i, line in enumerate(test_text):
    if len(line) == 0:
        print(f"Ligne {i} de test_text est vide.")

#en effectuant cette commande, j'ai vérifié quelles lignes étaient vides car les lignes 100 et 150 n'affichaient rien
#cela semble être "normal"


# In[12]:


#remplacement des mots par leurs représentations numériques (étape 1.5)

train_text_int=[]
test_text_int=[]

#pour le texte d'entraînement
for line in train_text:
    temp_line=[]
    tokens=line.split()
    for token in tokens:
        temp_line.append(word2idx.get(token,0))
    train_text_int.append(temp_line)
    
#pour le texte de test
for line in test_text:
    temp_line=[]
    tokens=line.split()
    for token in tokens:
        temp_line.append(word2idx.get(token,0))
    test_text_int.append(temp_line)
    
#affichage de quelques lignes pour comparer
for i in range(50, 201, 50):
    print(f"Ligne {i} de train_text_int:")
    print(train_text_int[i])
    print(f"Ligne {i} de train_text:")
    print(train_text[i])
    print("\n")


# In[16]:


#Exercice 2


# In[17]:


#création des représentations matricielles (étape 2.1)

V=len(word2idx)

#création des amtrices A0 et A1
A0 = np.ones((V,V))
A1 = np.ones((V,V))

#création des listes pi0 et pi1
pi0=np.ones(V)
pi1=np.ones(V)


# In[18]:


#extraction de caractéristiques stochastiques (étape 2.2)

def compute_counts(text_as_int, A, pi):
    #on parcourt chaque ligne de text_as_int qui représente une phrase
    for sentence in text_as_int:
        #pour chaque paire de mots dans la phrase
        for i in range(1, len(sentence)):
            #on compte les transitions entre mots et les ajoute à la matrice A
            A[sentence[i-1], sentence[i]] +=1
        #on compte les mots qui apparaissent en début de phrase et on les ajoute à la liste pi
        pi[sentence[0]] +=1
        
compute_counts([t for t, y in zip(train_text_int, Ytrain) if y == 0], A0, pi0)
#ici, l'appel de la fonction en compréhension permet de sélectionner les textes d'entraînement associés à l'étiquette 0 et les passe à la fonction avec la matruce A0 et la liste pi0 en arguments


# In[20]:


#normalisation (étape 2.3)

def normalize_matrix(A):
    for i in range(len(A)):
        row_sum=np.sum(A[i])
        A[i]=A[i]/row_sum
    return A

def normalize_pi(pi):
    pi_sum=np.sum(pi)
    pi = pi/pi_sum
    return pi

A0_normalized=normalize_matrix(A0)
A1_normalized=normalize_matrix(A1)
pi0_normalized = normalize_pi(pi0)
pi1_normalized = normalize_pi(pi1)


# In[21]:


#propriété log (étape 2.4)

logA0 = np.log(A0_normalized)
logA1 = np.log(A1_normalized)
logpi0 = np.log(pi0_normalized)
logpi1 = np.log(pi1_normalized)


# In[22]:


#priors (étape 2.5)

#nombre d'éléments
count0 = sum(1 for label in Ytrain if label==0)
count1 = sum(1 for label in Ytrain if label==1)
total=len(Ytrain)

#calcul des priors
p0= count0/total
p1= count1/total

#calcul des logarithmes des priors
logp0=np.log(p0)
logp1=np.log(p1)


# In[23]:


#Exercice 3


# In[25]:


class Classifier:
    def __init__(self, logAs, logpis, logpriors):
        self.logAs = logAs
        self.logpis = logpis
        self.logpriors = logpriors
    
    def compute_log_likelihood(self, input_, class_):
        log_likelihood = self.logpis[class_]
        for i in range(1, len(input_)):
            log_likelihood += self.logAs[class_][input_[i-1]][input_[i]]
        return log_likelihood
    
    def predict(self, inputs):
        predictions = []
        for input_ in inputs:
            max_log_likelihood = float('-inf')
            predicted_class = None
            for class_ in range(len(self.logpriors)):
                log_likelihood = self.compute_log_likelihood(input_, class_)
                if log_likelihood > max_log_likelihood:
                    max_log_likelihood = log_likelihood
                    predicted_class = class_
            predictions.append(predicted_class)
        return predictions

#instanciation du Classifier avec les listes correctement remplies
classifier = Classifier(logAs, logpis, logpriors)

#prédiction des étiquettes pour les données de test
predictions = classifier.predict(test_text_int)

#affichage des prédictions et des étiquettes réelles
for i in range(len(predictions)):
    print("Prédiction:", predictions[i], " - Étiquette réelle:", Ytest[i])


# In[ ]:




