#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:59:12 2024

@author: Yeleen
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv('spam.csv', encoding='ISO 8859 1')

# Sélectionner les colonnes qui ne commencent pas par "Unnamed"
df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

# Renommer les colonnes
df.columns = ['labels', 'data']

# Mapper les étiquettes 'ham' et 'spam' vers les valeurs numériques
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})

# Diviser les données en ensembles d'entraînement et de test
df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

# Extraction de caractéristiques avec TfidfVectorizer
tfidf = TfidfVectorizer(decode_error='ignore')
Xtrain_tfidf = tfidf.fit_transform(df_train['data'])
Xtest_tfidf = tfidf.transform(df_test['data'])

# Extraction de caractéristiques avec CountVectorizer
count_vectorizer = CountVectorizer()
Xtrain_count = count_vectorizer.fit_transform(df_train['data'])
Xtest_count = count_vectorizer.transform(df_test['data'])

# Entraîner et évaluer le modèle Naive Bayes avec TfidfVectorizer
model_NB_tfidf = MultinomialNB()
model_NB_tfidf.fit(Xtrain_tfidf, df_train['b_labels'])
precision_NB_tfidf = model_NB_tfidf.score(Xtest_tfidf, df_test['b_labels'])
print("Précision pour Naive Bayes avec TfidfVectorizer :", precision_NB_tfidf)

# Entraîner et évaluer le modèle AdaBoost avec TfidfVectorizer
model_AdaBoost_tfidf = AdaBoostClassifier()
model_AdaBoost_tfidf.fit(Xtrain_tfidf, df_train['b_labels'])
predictions_AdaBoost_tfidf = model_AdaBoost_tfidf.predict(Xtest_tfidf)
precision_AdaBoost_tfidf = accuracy_score(df_test['b_labels'], predictions_AdaBoost_tfidf)
print("Précision pour AdaBoost avec TfidfVectorizer :", precision_AdaBoost_tfidf)

# Entraîner et évaluer le modèle Naive Bayes avec CountVectorizer
model_NB_count = MultinomialNB()
model_NB_count.fit(Xtrain_count, df_train['b_labels'])
precision_NB_count = model_NB_count.score(Xtest_count, df_test['b_labels'])
print("Précision pour Naive Bayes avec CountVectorizer :", precision_NB_count)

# Entraîner et évaluer le modèle AdaBoost avec CountVectorizer
model_AdaBoost_count = AdaBoostClassifier()
model_AdaBoost_count.fit(Xtrain_count, df_train['b_labels'])
predictions_AdaBoost_count = model_AdaBoost_count.predict(Xtest_count)
precision_AdaBoost_count = accuracy_score(df_test['b_labels'], predictions_AdaBoost_count)
print("Précision pour AdaBoost avec CountVectorizer :", precision_AdaBoost_count)

def visualize(label):
    words = ''
    for msg in df[df['labels'] == label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

# Visualiser le nuage de mots pour les messages 'ham'
visualize('ham')

# Visualiser le nuage de mots pour les messages 'spam'
visualize('spam')


# 1. **Précision pour Naive Bayes avec TfidfVectorizer (0.9599)** :
#    - Ce résultat montre une précision relativement élevée, mais pas la plus élevée parmi les quatre combinaisons de modèle et de vectoriseur. Cela suggère que Naive Bayes fonctionne bien avec la représentation des termes par leur fréquence inverse de document (TF-IDF), mais il existe probablement des modèles ou des techniques de vectorisation qui pourraient mieux convenir à ces données.

# 2. **Précision pour AdaBoost avec TfidfVectorizer (0.9713)** :
#    - AdaBoost montre une amélioration par rapport à Naive Bayes avec TfidfVectorizer, ce qui indique que l'algorithme de boosting a été capable d'exploiter davantage d'informations du TF-IDF pour obtenir de meilleures performances. Cependant, il est toujours possible qu'il existe d'autres combinaisons de modèles et de vectoriseurs qui puissent surpasser cette précision.

# 3. **Précision pour Naive Bayes avec CountVectorizer (0.9821)** :
#    - Cette précision est la plus élevée parmi les quatre combinaisons. Cela suggère que Naive Bayes fonctionne très bien avec la représentation simple des termes par leur fréquence brute (CountVectorizer). Il est probable que le texte soit assez bien séparé sur la base de la fréquence des mots pour que Naive Bayes fonctionne efficacement avec cette méthode de vectorisation.

# 4. **Précision pour AdaBoost avec CountVectorizer (0.9629)** :
#    - Bien que cette précision soit inférieure à celle de Naive Bayes avec CountVectorizer, elle reste assez élevée. Cela montre que même avec une technique de vectorisation moins sophistiquée, AdaBoost peut encore obtenir de bonnes performances en exploitant les caractéristiques de base des données.

# En résumé, ces résultats suggèrent que Naive Bayes fonctionne particulièrement bien avec CountVectorizer, tandis qu'AdaBoost semble mieux exploiter les informations fournies par TfidfVectorizer. En tant que programmeur, je pourrais être enclin à explorer davantage ces deux combinaisons, ainsi que d'autres modèles et techniques de vectorisation, pour optimiser encore les performances du système de classification de texte.

#Avec la fonction visualize on obtient des images qui comme je les aient affichées nous donne des nuages de mot classé ham ou spam