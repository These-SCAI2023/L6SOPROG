#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.naive_bayes import MultinomialNB
import  pandas as pd
import numpy as np


# In[3]:


data = pd.read_csv('spambase.data').values
# Mélanger les données
np.random.shuffle(data)

# Séparer les caractéristiques (X) et les étiquettes (Y)
X = data[:, :-1]  # Toutes les lignes, toutes les colonnes sauf la dernière
Y = data[:, -1]   # Toutes les lignes, seulement la dernière colonne

# Afficher la forme des matrices X et Y pour vérification
print("Forme de X :", X.shape)
print("Forme de Y :", Y.shape)



# In[4]:


# Diviser les données en ensembles d'entraînement et d'evaluation
Xtrain = X[:-100]  
Ytrain = Y[:-100]  
Xtest = X[-100:]   
Ytest = Y[-100:]   

# Afficher les formes des ensembles d'entraînement et de test
print("Forme de Xtrain :", Xtrain.shape)
print("Forme de Ytrain :", Ytrain.shape)
print("Forme de Xtest :", Xtest.shape)
print("Forme de Ytest :", Ytest.shape)


# In[5]:


#charger un modele
#entrainement
model = MultinomialNB()#creer une instance
model.fit(Xtrain,Ytrain)#chargeons le modèle avec nos propres données d’entraînement


# In[6]:


#evaluation
precision = model.score(Xtest,Ytest)
print('Precision pour NB:', precision)


# In[8]:


#repeter
from sklearn.ensemble import AdaBoostClassifier
modelA = AdaBoostClassifier()
modelA.fit(Xtrain,Ytrain)
precisionA = modelA.score(Xtest,Ytest)
print('Precision pour Ada:',precisionA)


# In[ ]:




