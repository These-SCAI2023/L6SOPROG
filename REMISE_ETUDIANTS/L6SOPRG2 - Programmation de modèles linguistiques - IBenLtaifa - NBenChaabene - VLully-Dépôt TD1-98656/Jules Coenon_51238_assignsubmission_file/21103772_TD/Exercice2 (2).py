import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer



# Chargement du fichier csv
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Suppression des colonnes "unnamed2/3/4"
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# Permet de renommer les colonnes avec "inplace" qui permet de garder la même variable
df.rename(columns={'v1': 'labels', 'v2': 'data'}, inplace=True)

# Ajout de la colonne "b_labels" avec des valeurs numériques
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})


# Séparation des données en données d'entraînement et de test
X = df['data']
y = df['b_labels']

# Utilisation de train_test_split() afin de partitionner les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Création d'un objet TfidfVectorizer pour la transformation TF-IDF afin d'obtenir des vecteurs
tfidf = TfidfVectorizer(decode_error='ignore')
# Modification pour adapter TfidfVectorizer aux données d'entraînement et on transforme les données d'entraînement
X_train_vec = tfidf.fit_transform(X_train)
# Transformation des données de test en utilisant le TfidfVectorizer adapté aux données d'entraînement
X_test_vec = tfidf.transform(X_test)

# Création du modèle Naive Bayes et entraînement
model_NB = MultinomialNB()
model_NB.fit(X_train_vec, y_train)

# Création modèle AdaBoostClassifier et l'entraîner
model_AdaBoost = AdaBoostClassifier()
model_AdaBoost.fit(X_train_vec, y_train)

# Mesure de la précision du modèle sur les données de test
precision_test_NB = model_NB.score(X_test_vec, y_test)
print("Précision sur les données de test pour Naive Bayes:", precision_test_NB)

#mesure la précision du modèle sur les données d'entraînement et de test
precision_test_AdaBoost = model_AdaBoost.score(X_test_vec, y_test)
print("Précision sur les données de test pour AdaBoost:", precision_test_AdaBoost)


# On observe que le modèle adaboost a une meilleure précision que le modèle Naive Bayes


