#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer#a mesure de fréquence TF-IDF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# In[12]:


#traitement des donnes
df = pd.read_csv('spam.csv',encoding = 'latin-1')
#colonne= ['v2']
#df = df.drop(colonne, axis = 1)#supprimer les colonnes iutiles


# In[44]:


# Renommer les colonnes
df = df.rename(columns={'v1': 'labels', 'v2': 'data'})

# Ajouter une nouvelle colonne pour les étiquettes numériques
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})

# Afficher le DataFrame après les modifications
print(df.head())


# In[45]:


# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df['data'], df['b_labels'], test_size=0.3, random_state=42)

# Afficher les dimensions des ensembles d'entraînement et de test
print("Dimensions de l'ensemble d'entraînement X :", X_train.shape)
print("Dimensions de l'ensemble d'évaluation X :", X_test.shape)
print("Dimensions de l'ensemble d'entraînement y :", y_train.shape)
print("Dimensions de l'ensemble d'évaluation y :", y_test.shape)


# In[46]:


#entrainement du modele
#extraction de caracteristiques

tfidf = TfidfVectorizer(decode_error='ignore')#on ignore les erruers quand on utilise l'outil et ono utilise la mesure de frequence TF-IDF
Xtrain = tfidf.fit_transform(X_train)#on transform les datas pour entrainement à une matrice de TF-IDf
Xtest = tfidf.transform(X_test)#on transform les datas pour évaluation à une matrice TF-IDF


# In[48]:


#modele bayesien
model = MultinomialNB()
model.fit(Xtrain,y_train)  
precision = model.score(Xtest,y_test)
print('Precision pour NB:',precision)
#expliquer: ce résultat est plus haut que le premier, c'est-à-dire qu'il est plus précise. Je pense que c’est à cause de l’utilisation de vecteurs tfidf, qui mettent l’accent sur les poids d’importance des différents mots, donc c’est plus précis


# In[52]:


#modele Adaboost
from sklearn.ensemble import AdaBoostClassifier
modelA = AdaBoostClassifier()
modelA.fit(Xtrain, y_train)
precisionA = modelA.score(Xtest,y_test)
print("Precision pour Ada:",precisionA)


# In[ ]:





# In[ ]:




