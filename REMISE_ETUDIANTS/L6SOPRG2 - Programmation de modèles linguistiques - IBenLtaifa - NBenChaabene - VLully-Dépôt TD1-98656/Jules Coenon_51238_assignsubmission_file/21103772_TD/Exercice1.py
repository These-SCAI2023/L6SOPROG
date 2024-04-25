#!/usr/bin/env python
# coding: utf-8




from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier





#chargement du fichier spambase.data
data = pd.read_csv("spambase.data").values 
#on mélange les données de façon aléatoire
np.random.shuffle(data)


# Affectation à X de toutes les lignes de toutes les colonnes sauf la dernière
X = data[:, :-1]
# Affectation à Y de toutes les lignes de la dernière colonne
Y = data[:, -1]

# Données pour l'entrainement
Xtrain = X[:-100]
Ytrain = Y[:-100]

# Données pour le test
Xtest = X[-100:] 
Ytest = Y[-100:]

# Modèle MultinomialNB
mode_NBl = MultinomialNB()

# Entraînement du modèle avec les données d'entraînement
model_NB.fit(Xtrain, Ytrain)

# Mesure de la précision du modèle entraîné avec la fonction "score()"
precision_NB = model_NB.score(Xtest, Ytest)
print("Précision pour NB:", precision_NB)

# Utilisation du modèle AdaBoostClassifier
model_AdaBoost = AdaBoostClassifier()

# Entraînement du modèle avec les données d'entraînement
model_AdaBoost.fit(Xtrain, Ytrain)

# Calcul de la précision du modèle AdaBoost sur nos données test
precision_adaboost = model_AdaBoost.score(Xtest, Ytest)
print("Précision pour AdaBoost:", precision_adaboost)



