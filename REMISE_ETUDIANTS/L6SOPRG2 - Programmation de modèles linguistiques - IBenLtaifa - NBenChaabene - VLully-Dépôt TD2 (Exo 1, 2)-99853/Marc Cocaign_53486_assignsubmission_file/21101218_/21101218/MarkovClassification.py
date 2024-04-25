#!/usr/bin/env python
# coding: utf-8

# # # # Importation de modules

# In[25]:


import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split


# In[26]:


#conda install posix
#Afin de faire fonctionner la commande "!head"


# In[27]:


#test
get_ipython().system('head edgar_allan_poe.txt')
print("______")
get_ipython().system('head robert_frost.txt')


# # Main

# ### Lecture

# In[28]:


input_files = ["edgar_allan_poe.txt", "robert_frost.txt"]
input_texts, labels, f, label = [], [], [], []
for fichier in enumerate(input_files):
    f.append(fichier[1])
    label.append(fichier[0])
    
print(f, label, sep='\n')


# In[29]:




for fichier in f: #La variable f contient les noms des fichiers
    with open(fichier, "r") as file:
        #file = open(fichier, "r")
        print(file.read())
        for ligne in file:
            line = ligne.lower()
            a = ligne.rstrip(ligne[-1])
            b = ligne.translate(str.maketrans('', '', string.punctuation))
            input_texts.append(line)
            #labels.append(input_files[fichier])
        lecture = file.read()
        print(lecture)
        #fichier.close()


# In[30]:


input_files = ["edgar_allan_poe.txt", "robert_frost.txt"]
input_texts, labels = [], []
for label, f in enumerate(input_files):
    for l in open(f):
        #print(l)
        line = l.lower()
        line = line.rstrip()
        line = line.translate(str.maketrans('', '', string.punctuation))
        print(line)
        input_texts.append(line)
        labels.append(label)


# In[31]:


train_text, test_text, Ytrain, Ytest = train_test_split(input_texts, labels)

#x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.3,shuffle=True)


# In[32]:


len(Ytrain) #Nombre de lignes par classe
train_text[:5] #Les 5 premières classes du texte d'entrainement
Ytrain[:5] #La correspondance des 5 premières lignes avec les prédictions

print(595/(595+1783)) #On obtient en réalité une répartition de 25%/75%


# In[33]:


word2idx = {}
idx = 1
for l in train_text:
    tokens = l.split(" ")
    for m in tokens:
        if m not in word2idx:
            word2idx[m] = idx
        else:
            pass
        idx += 1  
        
#print(word2idx)
print(len(word2idx))


# In[34]:


train_text_int = []
test_text_int = []
for l in train_text:
    liste_temporaire = []
    tokens_train = l.split(" ")
    for m in tokens_train:
        liste_temporaire.append(word2idx[m])
    #print(liste_temporaire)
    train_text_int.append(liste_temporaire)
    test_text_int.append(liste_temporaire)
    
    
print(train_text_int, end = f"\n________________\n")    
print(train_text_int)


# In[35]:


#print(len(train_text_int), len(test_text_int))

for n in range(1783): #1783 = longueur des listes
    #print(train_text_int[n], test_text_int[n], sep="\n")
    if (train_text_int[n] == test_text_int[n]) == True:
        #print("True", end = " ")
        pass
    else:
        print("Non concordance")
        break

#Les deux listes sont les mêmes


# # Exercice 2: Entraînement du modèle

# In[36]:


#V = len(word2idx)
V = 20000 # word2idx est trop court
A0, A1 = np.ones((V, V)), np.ones((V, V))
pi0, pi1 = np.ones(V), np.ones(V)
print(V)


# In[37]:


def compute_counts(test_text_int, A, pi):
    for tokens in test_text_int:
        for idx in tokens:
            if idx is First:
                pi[idx] += 1
            else:    
                A[last_idx, idx] += 1


# In[38]:


A0,pi0= np.ones((V, V)),np.ones(V)

def compute_counts(text_as_int, A, pi):
    for ligne in text_as_int:
        for token in ligne:
            if token == ligne[0]: #vu que les mots ont tous un index unique
                pi[token] += 1
            else:
                A[ligne[-1], token] += 1
            
compute_counts(train_text_int,A0,pi0)
print(A0)
print(pi0)


# In[39]:


"""
A0, A1 = np.ones((V, V)), np.ones((V, V))
pi0, pi1 = np.ones(V), np.ones(V)
compute_counts([t for t, y in zip(train_text_int, Ytrain) if y == 0],A0,pi0)
#Cette ligne applique la fonction ci-dessus sur le corpus Poe, pour chaque mot du corpus d'entraînement 
print(A0)
print(pi0)
"""


# In[ ]:


#Erreur à cause de la valeur de 20000
for l in A0:
    print(l, len(l), sep=" ")
    for valeur in l:
        valeur = valeur / len(l)
        
for l in pi0:
    print(l, len(l), sep=" ")
    for valeur in l:
        valeur = valeur / len(l)


# In[ ]:


logA0 = np.log(A0)
logpi0 = np.log(pi0)
logA1 = np.log(A1)
logpi1 = np.log(pi1)


# In[ ]:


count0 = sum(Ytrain where label==0)
count1 = sum(Ytrain where label==1)
total = len(Ytrain)
p0=count0/total
p1=count1/total
logp0=np.log(p0)
logp1=np.log(p1)

