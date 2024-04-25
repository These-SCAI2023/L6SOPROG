#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:09:27 2024

@author: Yeleen
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Charger les données
data = pd.read_csv('spambase.data').values
np.random.shuffle(data)

# Séparer les caractéristiques (X) et les étiquettes (Y)
X = data[:, :-1]
Y = data[:, -1]

# Diviser les données en ensembles d'entraînement et de test
Xtrain = X[:-100]
Ytrain = Y[:-100]
Xtest = X[-100:]
Ytest = Y[-100:]

# Entraîner et évaluer le modèle Naive Bayes
model_NB = MultinomialNB()
model_NB.fit(Xtrain, Ytrain)
precision_NB = model_NB.score(Xtest, Ytest)
print("Précision pour Naive Bayes :", precision_NB)

# Entraîner et évaluer le modèle AdaBoost
model_AdaBoost = AdaBoostClassifier()
model_AdaBoost.fit(Xtrain, Ytrain)
predictions_AdaBoost = model_AdaBoost.predict(Xtest)
precision_AdaBoost = accuracy_score(Ytest, predictions_AdaBoost)
print("Précision pour AdaBoost :", precision_AdaBoost)

# Lorsque nous comparons les performances des deux modèles, Naive Bayes et AdaBoost, sur l'ensemble de test, nous remarquons une différence significative dans leurs précisions :

# 1. **Naive Bayes** : La précision obtenue est de 0,8. Cela signifie que le modèle Naive Bayes a correctement classé environ 80% des échantillons de l'ensemble de test.

# 2. **AdaBoost** : La précision obtenue est de 0,92. Cela indique que le modèle AdaBoost a correctement classé environ 92% des échantillons de l'ensemble de test.

# En conclusion :
# - **AdaBoost** a montré une meilleure performance par rapport à Naive Bayes sur cet ensemble de données spécifique. Cela peut être dû à la capacité d'AdaBoost à construire un modèle fort en combinant plusieurs modèles faibles, ce qui lui permet de mieux généraliser et de mieux s'adapter aux données.
# - Cependant, il est important de noter que les performances des modèles peuvent varier en fonction des caractéristiques des données et de la manière dont elles sont traitées. Il est donc recommandé de tester plusieurs modèles et de sélectionner celui qui donne les meilleures performances pour un problème spécifique.