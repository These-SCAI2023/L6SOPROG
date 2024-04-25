#!/usr/bin/env python
# coding: utf-8

# In[35]:


# Importation du modèle Naive Bayes multinomial depuis scikit-learn
from sklearn.naive_bayes import MultinomialNB
# Importation de pandas sous le nom pd pour manipulation de données
import pandas as pd
# Importation de numpy sous le nom np pour manipulation de tableaux
import numpy as np

# Chargement des données à partir d'un fichier CSV nommé 'spambase.data' et conversion en tableau NumPy
data = pd.read_csv('spambase.data').values
# Mélange aléatoire des lignes du tableau de données
np.random.shuffle(data)
# Affichage des données mélangées
print(data)

# Séparation des caractéristiques (X) et de la cible (y) à partir des données
X = data[:, :-1]  # Sélection de toutes les colonnes sauf la dernière pour X
y = data[:, -1]   # Sélection de la dernière colonne pour y

# Séparation des données en ensemble d'entraînement et ensemble de test
X_train = X[:-100]  # Les 100 dernières lignes sont réservées pour le test
Y_train = y[:-100]  # Les 100 dernières étiquettes correspondent à l'ensemble d'entraînement
X_test = X[-100:]   # Les 100 dernières lignes sont utilisées pour le test
Y_test = y[-100:]   # Les 100 dernières étiquettes correspondent à l'ensemble de test

# Initialisation d'un modèle Naive Bayes multinomial
model = MultinomialNB()
# Entraînement du modèle sur l'ensemble d'entraînement
model.fit(X_train, Y_train)
# Calcul de la précision du modèle sur l'ensemble de test
precision = model.score(X_test , Y_test)
# Affichage de la précision du modèle Naive Bayes multinomial
print("Précision pour NB:", precision)


# In[36]:


# Importation du modèle AdaBoostClassifier depuis scikit-learn
from sklearn.ensemble import AdaBoostClassifier
# Importation de la fonction accuracy_score pour évaluer les performances du modèle
from sklearn.metrics import accuracy_score
# Importation de pandas sous le nom pd pour manipulation de données
import pandas as pd
# Importation de numpy sous le nom np pour manipulation de tableaux
import numpy as np

# Chargement des données à partir d'un fichier CSV nommé 'spambase.data' et conversion en tableau NumPy
data = pd.read_csv('spambase.data').values
# Mélange aléatoire des lignes du tableau de données
np.random.shuffle(data)

# Séparation des caractéristiques (X) et de la cible (y) à partir des données
X = data[:, :-1]  # Sélection de toutes les colonnes sauf la dernière pour X
y = data[:, -1]   # Sélection de la dernière colonne pour y

# Séparation des données en ensemble d'entraînement et ensemble de test
X_train = X[:-100]  # Les 100 dernières lignes sont réservées pour le test
y_train = y[:-100]  # Les 100 dernières étiquettes correspondent à l'ensemble d'entraînement
X_test = X[-100:]   # Les 100 dernières lignes sont utilisées pour le test
y_test = y[-100:]   # Les 100 dernières étiquettes correspondent à l'ensemble de test

# Initialisation d'un modèle AdaBoostClassifier
model = AdaBoostClassifier()

# Entraînement du modèle sur l'ensemble d'entraînement
model.fit(X_train, y_train)

# Prédiction des étiquettes pour l'ensemble de test
y_pred = model.predict(X_test)

# Calcul de l'exactitude du modèle en comparant les étiquettes prédites avec les étiquettes réelles de l'ensemble de test
accuracy = accuracy_score(y_test, y_pred)

# Affichage de l'exactitude du modèle AdaBoostClassifier
print("Accuracy for AdaBoost Classifier:", accuracy)

