#!/usr/bin/env python
# coding: utf-8

# In[73]:


import numpy as np  # Importation de la bibliothèque numpy pour le support de tableaux et de matrices
import pandas as pd  # Importation de la bibliothèque pandas pour la manipulation des données tabulaires
from sklearn.feature_extraction.text import TfidfVectorizer  # Importation du vecteur de caractéristiques TF-IDF
from sklearn.model_selection import train_test_split  # Importation de la fonction de scission des données d'entraînement et de test
from sklearn.naive_bayes import MultinomialNB  # Importation du classificateur Naive Bayes multinomial

df = pd.read_csv('spam.csv', encoding='ISO-8859-1')  # Chargement des données à partir du fichier CSV dans un DataFrame pandas

df = df.drop(df.columns[2:], axis=1)  # Suppression des colonnes inutiles à partir de l'index 2 (inclus) jusqu'à la fin

print(df.head())  # Affichage des premières lignes du DataFrame pour vérification

df.rename(columns={'v1': 'labels', 'v2': 'data'}, inplace=True)  # Renommage des colonnes 'v1' et 'v2' en 'labels' et 'data'

df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})  # Création d'une nouvelle colonne 'b_labels' avec des valeurs binaires pour les étiquettes

print(df.head())  # Affichage des premières lignes du DataFrame mis à jour pour vérification


# In[74]:


from sklearn.model_selection import train_test_split  # Importation de la fonction de scission des données d'entraînement et de test
from sklearn.feature_extraction.text import TfidfVectorizer  # Importation du vecteur de caractéristiques TF-IDF
from sklearn.naive_bayes import MultinomialNB  # Importation du classificateur Naive Bayes multinomial
from sklearn.metrics import accuracy_score  # Importation de la fonction pour calculer la précision du modèle

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df['data'], df['labels'], test_size=0.3, random_state=42)

# Création d'une instance de TfidfVectorizer
tfidf = TfidfVectorizer(decode_error='ignore')

# Transformation des données d'entraînement en vecteurs TF-IDF
X_train_tfidf = tfidf.fit_transform(X_train)

# Transformation des données de test en vecteurs TF-IDF en utilisant les mêmes paramètres que ceux de l'entraînement
X_test_tfidf = tfidf.transform(X_test)

# Initialisation du classificateur Naive Bayes multinomial
nb_classifier = MultinomialNB()

# Entraînement du classificateur Naive Bayes multinomial sur les données d'entraînement TF-IDF
nb_classifier.fit(X_train_tfidf, y_train)

# Prédiction des étiquettes pour les données d'entraînement et de test
y_train_pred = nb_classifier.predict(X_train_tfidf)
y_test_pred = nb_classifier.predict(X_test_tfidf)

# Calcul de la précision du modèle sur les données d'entraînement et de test
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Affichage de la précision du modèle sur les ensembles d'entraînement et de test
print("Naive Bayes - Training Accuracy:", train_accuracy)
print("Naive Bayes - Testing Accuracy:", test_accuracy)


# In[78]:


from sklearn.ensemble import AdaBoostClassifier  # Importation du classificateur AdaBoost

# Initialisation du classificateur AdaBoost avec 50 estimateurs et une graine aléatoire fixée à 42
adaboost_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)

# Entraînement du classificateur AdaBoost sur les données d'entraînement TF-IDF
adaboost_classifier.fit(X_train_tfidf, y_train)

# Prédiction des étiquettes pour les données d'entraînement et de test à l'aide du classificateur AdaBoost
y_train_pred_ab = adaboost_classifier.predict(X_train_tfidf)
y_test_pred_ab = adaboost_classifier.predict(X_test_tfidf)

# Calcul de la précision du modèle AdaBoost sur les données d'entraînement et de test
train_accuracy_ab = accuracy_score(y_train, y_train_pred_ab)
test_accuracy_ab = accuracy_score(y_test, y_test_pred_ab)

# Affichage de la précision du modèle AdaBoost sur les ensembles d'entraînement et de test
print("AdaBoost - Training Accuracy:", train_accuracy_ab)
print("AdaBoost - Testing Accuracy:", test_accuracy_ab)


# In[86]:


from sklearn.feature_extraction.text import CountVectorizer  # Importation du vecteur de caractéristiques CountVectorizer


# In[87]:


import pandas as pd  # Importation de la bibliothèque pandas pour la manipulation des données tabulaires
import matplotlib.pyplot as plt  # Importation de la bibliothèque matplotlib pour la visualisation des données
from wordcloud import WordCloud  # Importation de la classe WordCloud pour générer un nuage de mots

def visualize(label):
    words = ''  # Initialisation d'une chaîne vide pour stocker les mots
    for msg in df[df['labels'] == label]['data']:  # Parcourir chaque message correspondant à l'étiquette spécifiée
        msg = msg.lower()  # Convertir le message en minuscules
        words += msg + ' '  # Ajouter les mots du message à la chaîne de mots
    wordcloud = WordCloud(width=600, height=400).generate(words)  # Créer un nuage de mots à partir des mots accumulés
    plt.imshow(wordcloud)  # Afficher le nuage de mots
    plt.axis('off')  # Désactiver les axes
    plt.show()  # Afficher le nuage de mots généré


# In[88]:


visualize('ham')  # Appel de la fonction visualize avec l'argument 'ham' pour générer un nuage de mots des messages étiquetés 'ham'


# In[89]:


visualize('spam')  # Appel de la fonction visualize avec l'argument 'spam' pour générer un nuage de mots des messages étiquetés 'spam'


# In[ ]:




