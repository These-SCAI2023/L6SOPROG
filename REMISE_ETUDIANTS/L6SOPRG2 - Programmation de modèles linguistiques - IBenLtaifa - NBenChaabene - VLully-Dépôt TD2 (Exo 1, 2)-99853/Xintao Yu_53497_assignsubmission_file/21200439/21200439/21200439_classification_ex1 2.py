#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importation des bibliothèques nécessaires
import numpy as np
import matplotlib .pyplot as plt
import string
from sklearn. model_selection import train_test_split


# In[6]:


get_ipython().system('head edgar_allan_poe.txt #A f f i c h e r l e s 10 prem i è r e s l i g n e s')
get_ipython().system('head robert_frost.txt')


# In[7]:


# Définir les fichiers sources
input_files = ['edgar_allan_poe.txt', 'robert_frost.txt']


# In[9]:


# Initialisation des listes pour le texte et les étiquettes
input_texts = []
labels = []


# In[43]:


# Lecture et prétraitement des textes
for index, filename in enumerate(input_files):

    with open(filename, 'r', encoding='utf-8') as file:
       
        for line in file:
           # Convertir en minuscules
            line = line.lower()
           # Supprimer les ponctuations
            line = line.rstrip().translate(str.maketrans('', '', string.punctuation))
           # Ajouter le texte et l'étiquette correspondante
            input_texts.append(line)
          
            labels.append(index)
            
# Afficher les premiers textes et leurs étiquettes
for text, label in zip(input_texts[:10], labels[:10]):
    print(f"Text: {text} \nLabel: {label}\n")


# In[15]:


# Division du corpus en ensembles d'entraînement et de test
train_text, test_text, Ytrain, Ytest = train_test_split(input_texts, labels, test_size=0.3, random_state=42)


# In[42]:


# Affichage des informations sur les corpus d'entraînement et de test
print("length de ytrain:", len(Ytrain))
print("length de ytest:", len(Ytest))
print("5 premiers lignes:", train_text[:5])
print("5 premier labels:", Ytrain[:5])


# In[41]:


# Création du dictionnaire de mots
word2idx = {'<unk>': 0}  
idx = 1  

# Construction du dictionnaire à partir du corpus d'entraînement
for text in train_text:

    tokens = text.split()

    for token in tokens:
       
        if token not in word2idx:
            word2idx[token] = idx
            idx += 1


# Afficher les premiers éléments du dictionnaire et la taille du vocabulaire
print(list(word2idx.items())[:100])  
print("la taille du vocabulaire:", len(word2idx))  


# In[19]:


train_text_int = []
test_text_int = []


# In[32]:


# Convertir les textes en séquences d'entiers
train_text_int = [[word2idx.get(token, word2idx['<unk>']) for token in text.split()] for text in train_text]


test_text_int = [[word2idx.get(token, word2idx['<unk>']) for token in text.split()] for text in test_text]


# In[33]:


# Afficher les conversions pour certaines lignes spécifiques
indexes = [49, 99, 149]  

for idx in indexes:
   
    if idx < len(train_text) and idx < len(train_text_int):
        print(f"train_text {idx + 1} eme ligne: {train_text[idx]}")
        print(f"train_text_int {idx + 1} eme ligne: {train_text_int[idx]}\n")
    else:
        print(f"index {idx + 1} depasse le range du corpus。\n")


# # ex2 

# In[39]:


# Calcul de la taille du vocabulaire V basé sur word2idx
V = len(word2idx)

# Création des matrices de transition A0 et A1, initialisées avec des 1 pour la suavisation de Laplace
A0 = np.ones((V, V))  # Matrice pour Edgar Allan Poe
A1 = np.ones((V, V))  # Matrice pour Robert Frost

# Création des vecteurs de probabilité initiale pi0 et pi1, également initialisés avec des 1
pi0 = np.ones(V)  # Vecteur pour Edgar Allan Poe
pi1 = np.ones(V)  # Vecteur pour Robert Frost

# Affichage des dimensions pour vérification
print("Dimension de A0:", A0.shape)
print("Dimension de A1:", A1.shape)
print("Dimension de pi0:", pi0.shape)
print("Dimension de pi1:", pi1.shape)




# In[25]:


# Fonction pour calculer les comptes pour A et pi
def compute_counts(text_as_int, A, pi):
    last_idx = None
    for tokens in text_as_int:
        for i, idx in enumerate(tokens):
            if i == 0:  
                pi[idx] += 1
            if last_idx is not None:  
                A[last_idx, idx] += 1
            last_idx = idx  
        last_idx = None  

# Appliquer compute_counts pour les deux auteurs
compute_counts([t for t, y in zip(train_text_int, Ytrain) if y == 0], A0, pi0)
compute_counts([t for t, y in zip(train_text_int, Ytrain) if y == 1], A1, pi1)


# In[26]:


# Fonction pour normaliser A et pi
def normalize(A, pi):
    
    for i in range(len(A)):
        A[i] /= A[i].sum()

    pi /= pi.sum()
    
# Normalisation des matrices et vecteurs
normalize(A0, pi0)
normalize(A1, pi1)


# In[27]:


# Calcul du logarithme des matrices et vecteurs pour éviter les problèmes de sous-débordement numérique
logA0 = np.log(A0)
logA1 = np.log(A1)
logpi0 = np.log(pi0)
logpi1 = np.log(pi1)


# In[38]:


# Calcul des probabilités a priori pour chaque classe
count0 = sum(1 for y in Ytrain if y == 0)
count1 = sum(1 for y in Ytrain if y == 1)
total = len(Ytrain)
p0 = count0 / total
p1 = count1 / total

# Calcul du logarithme des probabilités a priori
logp0 = np.log(p0)
logp1 = np.log(p1)



# In[ ]:




