# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:13:19 2024

"""
#imports :
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np

#préparation des données :
data = pd.read_csv('spambase.data').values
np.random.shuffle(data)

#séparation des caractéristiques des textes et des étiquettes :
X = []
for elt in data :
    X.append(elt)
    
for lmnt in X :
    np.delete(lmnt, -1)
        
Y = []
for elt in data :
    Y.append(elt[-1])
    
#division des données pour l'entraînement et l'évaluation :
Xtrain = X[0:-100]
Ytrain = Y[0:-100]

Xtest = X[-100:]
Ytest = Y[-100:]
    
#entraînement du modèle :
model = MultinomialNB()
model.fit(Xtrain, Ytrain)

precision = model.score(Xtest, Ytest)
print("Précision pour NB:", precision)

#Nous obtenons toujours un résultat de 1 avec le modèle AdaBoostClassifier bien que le premier modèle donne effectivement toujours des résultats différents:
model2 = AdaBoostClassifier()
model2.fit(Xtrain, Ytrain)

precision2 = model2.score(Xtest, Ytest)
print("Précision pour AdaBoost:", precision2)

#Comparaison des résultats : le premier modèle donne des résultats entre 0.8 et 0.9, tandis que le second donne toujours un résultat de 1 