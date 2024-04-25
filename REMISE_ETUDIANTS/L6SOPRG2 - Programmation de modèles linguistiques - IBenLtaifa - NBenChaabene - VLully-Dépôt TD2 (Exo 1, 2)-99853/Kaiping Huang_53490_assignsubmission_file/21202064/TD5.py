#!/usr/bin/env python
# coding: utf-8

# In[95]:


import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split


# In[96]:


get_ipython().system('head edgar_allan_poe.txt')


# In[97]:


get_ipython().system('head robert_frost.txt')


# In[98]:


input_files = ['edgar_allan_poe.txt', 'robert_frost.txt']


# In[99]:


import string

# Listes pour stocker les textes et les étiquettes
input_texts = []
labels = []

# Noms des fichiers et leurs étiquettes correspondantes
input_files = ['edgar_allan_poe.txt', 'robert_frost.txt']
file_labels = ['edgar_allan_poe', 'robert_frost']

# Boucle pour lire les fichiers
for label, file_name in zip(file_labels, input_files):
    with open(file_name, 'r') as file:
        # Lecture ligne par ligne du fichier
        for line in file:
            # Prétraitement
            line = line.lower()  # Conversion en minuscules
            line = line.rstrip()  # Suppression des espaces à droite
            line = line.translate(str.maketrans('', '', string.punctuation))  # Suppression de la ponctuation

            # Ajout du texte et de l'étiquette correspondante aux listes
            input_texts.append(line)
            labels.append(label)


# In[100]:


train_text, test_text, Ytrain, Ytest = train_test_split(input_texts, labels)


# In[101]:


len(Ytrain), len(Ytest) 
train_text [:5]
Ytrain [:5]


# In[102]:


"""
ces commandes préparent des données textuelles pour un apprentissage supervisé 
en divisant les données en ensembles d'entraînement et de test, 
et en effectuant un prétraitement de base sur les textes comme la conversion en minuscules 
et la suppression de la ponctuation.
"""


# In[103]:


from sklearn.model_selection import train_test_split

# Division du corpus
train_text, test_text, Ytrain, Ytest = train_test_split(input_texts, labels, test_size=0.3, random_state=42)

# Vérification des tailles des ensembles d'entraînement et de test
print("Taille de Ytrain :", len(Ytrain))
print("Taille de Ytest :", len(Ytest))

# Affichage des cinq premières lignes du corpus d'entraînement
print("\nCinq premières lignes du corpus d'entraînement :\n", train_text[:5])

# Affichage des cinq premières étiquettes du corpus d'entraînement
print("\nCinq premières étiquettes du corpus d'entraînement :\n", Ytrain[:5])


# In[104]:


word2idx = {'<unk>': 0}  # Initialisation avec un index pour les mots inconnus
idx = 1  # Commencer à compter les index à partir de 1

# Parcours de chaque mot dans le vocabulaire
for word in set(train_text + test_text):
    if word not in word2idx:
        word2idx[word] = idx  # Associer le mot à son index
        idx += 1  # Incrémenter l'index pour le prochain mot unique

print("Nombre total de mots dans le vocabulaire :", len(word2idx))


# In[105]:


word2idx = {'<unk>': 0}  # Initialisation avec un index pour les mots inconnus
idx = 1  # Commencer à compter les index à partir de 1

# Boucle pour parcourir les lignes du corpus d'entraînement
for line in train_text:
    # Découpage de la ligne en mots
    tokens = line.split()
    # Parcours de chaque mot dans la ligne
    for token in tokens:
        # Validation si le mot existe déjà dans word2idx
        if token not in word2idx:
            word2idx[token] = idx  # Ajout du mot au dictionnaire avec son index
            idx += 1  # Incrémentation de l'index pour le prochain mot unique

# Affichage du contenu du dictionnaire word2idx
print("Contenu du dictionnaire word2idx :", word2idx)
# Affichage de la taille du vocabulaire
print("Taille du vocabulaire :", len(word2idx))


# In[106]:


# Initialisation des listes pour stocker les représentations numériques
train_text_int = []
test_text_int = []

# Convertir les mots en représentations numériques dans le corpus d'entraînement
for line in train_text:
    temp_int_line = []  # Liste temporaire pour stocker les représentations numériques de chaque mot de la ligne
    # Découpage de la ligne en mots
    tokens = line.split()
    if tokens:
        # Parcours de chaque mot dans la ligne
        for token in tokens:
            # Recherche de la représentation numérique du mot dans le dictionnaire word2idx
            if token in word2idx:
                temp_int_line.append(word2idx[token])  # Ajout de la représentation numérique à la liste temporaire
            else:
                temp_int_line.append(word2idx['<unk>'])  # Ajout de l'index pour les mots inconnus
        # Ajout de la liste temporaire à train_text_int
        train_text_int.append(temp_int_line)

# Convertir les mots en représentations numériques dans le corpus d'évaluation
for line in test_text:
    temp_int_line = []  # Liste temporaire pour stocker les représentations numériques de chaque mot de la ligne
    # Découpage de la ligne en mots
    tokens = line.split()
    if tokens:
        # Parcours de chaque mot dans la ligne
        for token in tokens:
            # Recherche de la représentation numérique du mot dans le dictionnaire word2idx
            if token in word2idx:
                temp_int_line.append(word2idx[token])  # Ajout de la représentation numérique à la liste temporaire
            else:
                temp_int_line.append(word2idx['<unk>'])  # Ajout de l'index pour les mots inconnus
        # Ajout de la liste temporaire à test_text_int
        test_text_int.append(temp_int_line)

# Comparaison des lignes 50, 100, 150, etc. de train_text_int avec les mêmes lignes de train_text
indices_to_check = [50, 100, 150]  # Indices des lignes à vérifier
for idx in indices_to_check:
    print("Ligne", idx, "de train_text_int :", train_text_int[idx])
    print("Ligne", idx, "de train_text :", train_text[idx])
    print()


# In[107]:


import numpy as np

# Création des matrices A0 et A1 et des listes pi0 et pi1
V = len(word2idx)  # Taille du vocabulaire
A0 = np.ones((V, V))  # Matrice pour gérer les probabilités de transition pour le corpus Poe
A1 = np.ones((V, V))  # Matrice pour gérer les probabilités de transition pour le corpus Frost
pi0 = np.ones(V)  # Liste pour gérer les probabilités d'apparition en début de phrase pour le corpus Poe
pi1 = np.ones(V)  # Liste pour gérer les probabilités d'apparition en début de phrase pour le corpus Frost

# Vérification des dimensions des matrices et des listes
print("Dimensions de la matrice A0 :", A0.shape)
print("Dimensions de la matrice A1 :", A1.shape)
print("Dimensions de la liste pi0 :", pi0.shape)
print("Dimensions de la liste pi1 :", pi1.shape)


# In[108]:


def compute_counts(text_as_int, A, pi):
    for line in text_as_int:
        for idx, word_idx in enumerate(line):
            if idx == 0:  # Premier mot de la phrase
                pi[word_idx] += 1
            else:
                A[line[idx-1], word_idx] += 1


# In[109]:


compute_counts([t for t, y in zip(train_text_int, Ytrain) if y == 0], A0, pi0)


# In[110]:


"""
La ligne de code compute_counts([t for t, y in zip(train_text_int, Ytrain) if y == 0], A0, pi0) 
appelle la fonction compute_counts() 
en passant seulement les lignes du corpus 
où l'étiquette est égale à zéro (c'est-à-dire pour le corpus Poe), 
et utilise les matrices A0 et la liste pi0 pour stocker les résultats de comptage.
"""


# In[111]:


def normalize_matrix(A):
    for i in range(A.shape[0]):
        row_sum = np.sum(A[i, :])
        if row_sum != 0:  # Éviter une division par zéro
            A[i, :] /= row_sum

def normalize_list(pi):
    total_sum = np.sum(pi)
    if total_sum != 0:  # Éviter une division par zéro
        pi /= total_sum

# Normalisation de la matrice A0
normalize_matrix(A0)
# Normalisation de la matrice A1
normalize_matrix(A1)
# Normalisation de la liste pi0
normalize_list(pi0)
# Normalisation de la liste pi1
normalize_list(pi1)


# In[112]:


# Calcul des logarithmes des valeurs dans la matrice A0 et la liste pi0
logA0 = np.log(A0)
logpi0 = np.log(pi0)

# Calcul des logarithmes des valeurs dans la matrice A1 et la liste pi1
logA1 = np.log(A1)
logpi1 = np.log(pi1)


# In[113]:


# Nombre d'éléments annotés avec label=0
count0 = np.sum(Ytrain == 0)
# Nombre d'éléments annotés avec label=1
count1 = np.sum(Ytrain == 1)
total = len(Ytrain)
# Calcul des priors pour chaque classe
p0 = count0 / total
p1 = count1 / total
# Calcul des logarithmes des priors
logp0 = np.log(p0)
logp1 = np.log(p1)


# In[ ]:




