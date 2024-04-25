#!/usr/bin/env python
# coding: utf-8

# In[13]:


from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier


# In[5]:


#chargement du fichier spambase.data
data = pd.read_csv("spambase.data").values 
#on mélange les données de façon aléatoire
np.random.shuffle(data)


# In[8]:


#on affecte à X toutes les lignes de toutes les colonnes sauf la dernière
X = data[:, :-1]
#on affecte à Y toutes les lignes de la dernière colonne
Y = data[:, -1]


# In[9]:


#on divise les données en ensembles d'entraînement et de test
Xtrain = X[:-100] #toutes les lignes sauf les 100 dernières
Ytrain = Y[:-100]
Xtest = X[-100:] #les 100 dernières lignes
Ytest = Y[-100:]


# In[10]:


#chargement du modèle MultinomialNB
model = MultinomialNB()
#entraînement du modèle avec les données d'entraînement
model.fit(Xtrain, Ytrain)


# In[12]:


#mesure de la précision du modèle entraîné avec la fonction "score()"
precision_NB = model.score(Xtest, Ytest)
print("Précision pour NB:", precision_NB)


# In[14]:


#chargement du modèle AdaBoostClassifier
model_adaboost = AdaBoostClassifier()
#entraînement du modèle avec les données d'entraînement
model_adaboost.fit(Xtrain, Ytrain)
#enfin, on mesure la précision du modèle AdaBoost sur nos données test
precision_adaboost = model_adaboost.score(Xtest, Ytest)
print("Précision pour AdaBoost:", precision_adaboost)


# In[ ]:


#AdaBoost semble donc être plus précis !

