#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
import pandas as pd
from sklearn. feature_extraction .text import TfidfVectorizer
from sklearn. model_selection import train_test_split
from sklearn. naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


# In[59]:


# Charger le jeu de données spam
df = pd.read_csv('./spam.csv', encoding='ISO-8859-1')


# In[60]:


# Supprimer les colonnes non nécessaires
df = df.drop(df.columns[2:], axis=1)

# Renommer les colonnes pour une meilleure clarté
df.columns = ['labels', 'data'] 

# Mapper les étiquettes textuelles à des étiquettes numériques
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1}) 


# In[61]:


# Séparer le jeu de données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df['data'], df['b_labels'], test_size=0.3, random_state=42)


# In[62]:


# Initialiser le vectorisateur TF-IDF pour convertir le texte en un vecteur de nombres
tfidf = TfidfVectorizer(decode_error='ignore')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)




# In[63]:


# Initialiser et entraîner le classificateur Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Évaluer la précision sur les ensembles d'entraînement et de test
nb_train_Précision = nb_classifier.score(X_train_tfidf, y_train)
nb_test_Précision = nb_classifier.score(X_test_tfidf, y_test)

# Afficher la précision du classificateur Naive Bayes
print(f"Précision pour nb_train: {nb_train_Précision}")
print(f"Précision pour nb_test: {nb_test_Précision}")




# In[64]:


# Initialiser et entraîner le classificateur AdaBoost
ada_classifier = AdaBoostClassifier()
ada_classifier.fit(X_train_tfidf, y_train)

# Évaluer la précision sur les ensembles d'entraînement et de test
ada_train_Précision = ada_classifier.score(X_train_tfidf, y_train)
ada_test_Précision = ada_classifier.score(X_test_tfidf, y_test)

# Afficher la précision du classificateur AdaBoost
print(f"Précision pour ada_train: {ada_train_Précision}")
print(f"Précision pour ada_test: {ada_test_Précision}")


# # bonus

# In[66]:


pip install wordcloud


# In[72]:


from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[73]:


# Charger le jeu de données spam
df = pd.read_csv('./spam.csv', encoding='ISO-8859-1')
# Supprimer les colonnes non nécessaires et renommer les colonnes restantes
df = df.drop(df.columns[2:], axis=1)
df.columns = ['labels', 'data']


# In[74]:


# Mapper les étiquettes textuelles à des étiquettes numériques
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})


# In[75]:


# 2.5.1 Utilisation de CountVectorizer pour la conversion de texte en vecteurs d'occurrences
count_vectorizer = CountVectorizer(decode_error='ignore')
X_counts = count_vectorizer.fit_transform(df['data'])


# In[76]:


# Séparation des données en ensembles d'entraînement et de test
X_train_counts, X_test_counts, y_train, y_test = train_test_split(X_counts, df['b_labels'], test_size=0.3, random_state=42)


# In[77]:


# Entraîner le classificateur Naive Bayes avec CountVectorizer
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_counts, y_train)


# In[78]:


# Évaluer la précision du modèle sur l'ensemble de test
precision = nb_classifier.score(X_test_counts, y_test)
print(f"Précision avec CountVectorizer: {precision}")


# In[79]:


def visualize(label):
    words = ''
    # Concaténer tous les textes des messages correspondant au label donné
    for msg in df[df['labels'] == label]['data']:
        msg = msg.lower()  # Convertir en minuscules
        words += msg + ' '
    # Générer le WordCloud à partir des textes concaténés
    wordcloud = WordCloud(width=600, height=400).generate(words)
    # Afficher le WordCloud
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


# In[80]:


# Test de la fonction visualize pour 'ham' et 'spam'
visualize('ham')
visualize('spam')

