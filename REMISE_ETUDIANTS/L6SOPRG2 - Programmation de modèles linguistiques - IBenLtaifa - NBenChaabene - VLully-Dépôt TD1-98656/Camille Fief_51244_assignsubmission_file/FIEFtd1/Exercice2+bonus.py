#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# In[15]:


#on charge les données du fichier csv
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')


# In[16]:


#on affiche les premières lignes du DataFrame pour explorer sa structure
print(df.head())


# In[17]:


print(df.columns) #pour vérifier les titres des colonnes

#on supprime les colonnes "unnamed2/3/4" car elles sont vides
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)


# In[18]:


#on renomme les colonnes
df.rename(columns={'v1': 'labels', 'v2': 'data'}, inplace=True)
#on ajoute une nouvelle colonne "b_labels" avec les étiquettes numériques
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
#on affiche les premières lignes afin de vérifier nos modifications
print(df.head())


# In[34]:


#on divise les données en ensemble d'entraînement et de test
X = df['data']  # caractéristiques (texte)
y = df['b_labels']  #étiquettes (0 pour ham, 1 pour spam)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#création objet TfidfVectorizer pour la transformation TF-IDF
tfidf = TfidfVectorizer(decode_error='ignore')

#on adapte TfidfVectorizer aux données d'entraînement et on transforme les données d'entraînement
X_train_tfidf = tfidf.fit_transform(X_train)

#on transforme les données de test en utilisant le TfidfVectorizer adapté aux données d'entraînement
X_test_tfidf = tfidf.transform(X_test)

#création du modèle Naive Bayes (MultinomialNB) et entraînement
model_nb = MultinomialNB()
model_nb.fit(X_train_tfidf, y_train)

#mesure de la précision du modèle sur les données de test
precision_test_nb = model_nb.score(X_test_tfidf, y_test)
print("Précision sur les données de test pour Naive Bayes:", precision_test_nb)


# In[35]:


from sklearn.ensemble import AdaBoostClassifier

#création modèle AdaBoostClassifier et l'entraîner
model_adaboost = AdaBoostClassifier()
model_adaboost.fit(X_train_tfidf, y_train)

#mesure la précision du modèle sur les données d'entraînement et de test
precision_test_adaboost = model_adaboost.score(X_test_tfidf, y_test)
print("Précision sur les données de test pour AdaBoost:", precision_test_adaboost)


# In[ ]:


#comme dans l'exercice 1, le modèle adaboost a une meilleure précision (même si légère) que le modèle Naive Bayes
#cette différence peut être dûe par exemple à la complexité du modèle adaboost par rapport à l'autre


# In[36]:


from sklearn.feature_extraction.text import CountVectorizer

#création objet CountVectorizer pour la transformation basée sur les occurrences
count_vectorizer = CountVectorizer()

#on adapte le CountVectorizer aux données d'entraînement et transformer les données d'entraînement
X_train_count = count_vectorizer.fit_transform(X_train)

#on transforme les données de test en utilisant le CountVectorizer adapté aux données d'entraînement
X_test_count = count_vectorizer.transform(X_test)


# In[37]:


#modèle Naive Bayes (MultinomialNB) et l'entraîner
model_nb_count = MultinomialNB()
model_nb_count.fit(X_train_count, y_train)

#calcule de la précision sur les données de test pour Naive Bayes avec CountVectorizer
precision_test_nb_count = model_nb_count.score(X_test_count, y_test)

print("Précision sur les données de test pour Naive Bayes avec CountVectorizer:", precision_test_nb_count)


# In[38]:


#modèle AdaBoostClassifier et l'entraîner
model_adaboost_count = AdaBoostClassifier()
model_adaboost_count.fit(X_train_count, y_train)

#calcule de la précision sur les données de test pour AdaBoost avec CountVectorizer
precision_test_adaboost_count = model_adaboost_count.score(X_test_count, y_test)

print("Précision sur les données de test pour AdaBoost avec CountVectorizer:", precision_test_adaboost_count)


# In[39]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

def visualize(label):
    words = ''
    # Récupérer les messages correspondant au label donné dans le DataFrame
    for msg in df[df['labels'] == label]['data']:
        # Convertir le message en minuscules
        msg = msg.lower()
        # Concaténer les mots du message
        words += msg + ' '

    # Générer le nuage de mots
    wordcloud = WordCloud(width=600, height=400).generate(words)
    
    # Afficher le nuage de mots
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

# Test de la fonction visualize pour le label 'ham'
visualize('ham')

# Test de la fonction visualize pour le label 'spam'
visualize('spam')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:





# In[ ]:





# In[ ]:





# In[ ]:




