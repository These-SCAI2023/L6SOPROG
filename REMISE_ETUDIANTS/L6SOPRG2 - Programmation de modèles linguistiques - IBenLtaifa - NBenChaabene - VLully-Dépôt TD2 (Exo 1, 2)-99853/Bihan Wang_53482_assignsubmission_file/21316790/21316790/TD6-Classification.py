#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as ny
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split


# In[4]:


get_ipython().system('head edgar_allan_poe.txt#afficher les 10 premieres lignes')
get_ipython().system('head robert_frost.txt')


# In[30]:


input_files = []
import glob
for chemin in glob.glob('*.txt'):
    filename = chemin
    #print(filename)
    input_files.append(filename)
print(input_files)


# In[32]:


input_texts = []
labels = []

for label, f in enumerate(input_files):
    print("Index :", label)
    print("Nom de fichier :", f)
    
    with open(f,'r')as file:
        for line in file:
            line = line.lower()
            line = line.rstrip()# Supprimer le dernier caractère à droite (retour à la ligne)
            line = line.translate(str.maketrans('', '', string.punctuation))# Supprimer les signes de ponctuation
        
            input_texts.append(line)
            labels.append(label)
print("input texts:",input_texts)
print("label:", labels)


# In[56]:


train_text, test_text, Ytrain, Ytest = train_test_split(input_texts,labels,test_size = 0.3,random_state = 42)

len(Ytrain),len(Ytest)
train_text[:5]
Ytrain[:5]


# In[61]:


idx = 1
word2idx = {'<unk>': 0}  # Initialisation avec une représentation pour les mots inconnus

# Parcourir chaque texte dans train_text pour construire le vocabulaire
for text in train_text:
    words = text.split()
    for word in words:
        if word not in word2idx:
            # Assigner un identifiant unique au mot et l'ajouter à word2idx
            word2idx[word] = idx
            # Incrémenter l'index pour le prochain mot unique
            idx += 1
print(word2idx)
print("la taille de word2idx:",len(word2idx))


# In[74]:


#convert data into integer format(valeur numerique)
train_text_int = []
test_text_int = []

for line in train_text:
    representations1 = []# Initialiser une liste temporaire pour stocker les représentations numériques des mots dans cette ligne
    words = line.split()
    for word in words:
        if word in word2idx:
            representations1.append(word2idx[word])
        else:
            representations1.append(word2idx['<unk>'])# Si le mot n'est pas dans word2idx, ajouter la représentation du mot inconnu
    train_text_int.append(representations1)

for line in test_text:
    representations2 = []
    words = line.split()
    for word in words:
        if word in word2idx:
            representations2.append(word2idx[word])
        else:
            representations2.append(word2idx['<unk>'])
    test_text_int.append(representations2)
print("train text int:",train_text_int[50],train_text_int[100],train_text_int[150])
print("train text:",train_text[50],train_text[100],train_text[150])


# In[76]:


import numpy as np

# la taille V égale à la longueur du vocabulaire
V = len(word2idx)

# Créer les matrices A0 et A1 avec des dimensions V x V
A0 = np.ones((V, V))
A1 = np.ones((V, V))

# Créer les listes pi0 et pi1 avec une longueur égale à V
pi0 = np.ones(V)
pi1 = np.ones(V)

#print("Dimensions de A0 :", A0.shape)
#print("Dimensions de A1 :", A1.shape)
#print("Dimensions de pi0 :", pi0.shape)
#print("Dimensions de pi1 :", pi1.shape)


# In[80]:


def compute_counts(text_as_int, A, pi):
    #Compute counts for transition matrix A and initial probabilities pi.
    #text_as_int (list): List of lists containing integer representations of words.
    #A (numpy.ndarray): Transition matrix.
    #pi (list): List of initial probabilities.

    # Parcourir chaque séquence dans text_as_int
    for sequence in text_as_int:
        # Initialiser l'index précédent à None
        prev_idx = None
        
        # Parcourir chaque indice dans la séquence
        for idx in sequence:
            # Si prev_idx est None, cela signifie que c'est le premier mot de la phrase
            if prev_idx is None:
                # Augmenter le compteur pour ce mot dans la liste pi
                pi[idx] += 1
            else:
                # Si prev_idx n'est pas None, c'est le dernier mot de la phrase précédente, alors compter la transition
                A[prev_idx, idx] += 1
            
            # Mettre à jour prev_idx avec l'indice actuel pour le prochain itération
            prev_idx = idx
compute_counts([t for t,y in zip(train_text_int,Ytrain)if y == 0],A0,pi0)


# In[85]:


#normalisation: chaque ligne de la matrice etre egale a 1
for A in A0,A1:    
    for i in range(len(A)):#chaque ligne
        somme = np.sum(A[i])
        # Parcourir chaque élément de la ligne
        for j in range(len(A[i])):
            A[i][j]/=somme
for pi in pi0,pi1:
    pi_sum = np.sum(pi)
    for i in range(len(pi)):
        pi[i] /= pi_sum


# In[86]:


#log
logA0 = np.log(A0)
logpi0 = np.log(pi0)

logA1 = np.log(A1)
logpi1 = np.log(pi1)


# In[87]:


# Nb d’éléments annotés avec le label = 0
count0 = sum(1 for label in Ytrain if label == 0)

# Nb d’éléments annotés avec le label = 1
count1 = sum(1 for label in Ytrain if label == 1)

# Total des éléments dans Ytrain
total = len(Ytrain)

# Calcul des priors pour chaque classe
p0 = count0 / total
p1 = count1 / total

logp0 = np.log(p0)
logp1 = np.log(p1)


# In[ ]:




