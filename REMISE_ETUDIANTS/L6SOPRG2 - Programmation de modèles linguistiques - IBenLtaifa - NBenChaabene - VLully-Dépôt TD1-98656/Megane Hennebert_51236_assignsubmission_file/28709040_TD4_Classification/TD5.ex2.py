# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:05:22 2024

"""
#imports :
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
# from wordcloud import WordCloud
import matplotlib as plt

# #définition pour la partie bonus :
# def visualize(label) :
#     words= ""
#     for msg in df[df["labels"] == label]["data"]:
#         msg = msg.lower()
#         words += msg + " "
#     wordcloud = WordCloud(width = 600, height = 400).generate(words)
#     plt.imshow(wordcloud)
#     plt.axis("off")
#     plt.show()

#chargement des données :
df = pd.read_csv("spam.csv", encoding= "ISO 8859 1")

#suppression des colonnes inutiles :
df = df.drop("Unnamed: 3", axis= 1)
df = df.drop("Unnamed: 4", axis= 1)

#changement des noms de colonnes :
df = df.rename(columns={"v1": "labels"})
df = df.rename(columns={"v2": "data"})

#ajout de la colonne b_labels en fonction des données de la colonne "labels" :------------------------------
b_labels = []
for lmnt in df["labels"] :
    b_labels.append(lmnt)
#configuration des données de b_labels en fonction de celles de "labels" :
for plc, elt in enumerate(b_labels) :
    if elt == "ham" :
        b_labels[plc] = "0"
    elif elt == "spam" :
        b_labels[plc] = "1"

df["b_labels"] = b_labels
#-----------------------------------------------------------------------------------------------------------

#Entraînement du modèle :
df_train, df_test, Y_train, Y_test = train_test_split(df["data"], df["b_labels"], test_size = 0.3)

tfidf = TfidfVectorizer(decode_error = "ignore")
Xtrain = tfidf.fit_transform(df_train)# nous changeons les valeurs de df_train pour qu'elles correspondent à notre test
Xtest = tfidf.transform(df_test)# //

#entraînement du modèle :
model = MultinomialNB()
model.fit(Xtrain, Y_train)

precision = model.score(Xtest, Y_test)
print("Précision pour NB:", precision)

model2 = AdaBoostClassifier()
model2.fit(Xtrain, Y_train)

precision2 = model2.score(Xtest, Y_test)
print("Précision pour AdaBoost:", precision2)

#Nous obtenons des résultats variant entre de 0.94 et 0.96 pour NB, entre de 0.95 et 0.97 pour AdaBoost

#Bonus