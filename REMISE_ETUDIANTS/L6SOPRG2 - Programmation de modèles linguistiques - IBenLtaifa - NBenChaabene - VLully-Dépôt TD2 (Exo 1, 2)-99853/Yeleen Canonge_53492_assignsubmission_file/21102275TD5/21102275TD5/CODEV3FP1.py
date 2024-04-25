#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:39:24 2024

@author: Yeleen
"""
# Import des bibliothèques nécessaires
from sklearn.model_selection import train_test_split
import numpy as np


# Étape 1: Téléchargement
input_files = ["edgar_allan_poe.txt", "robert_frost.txt"]

# Étape 2: Lecture de fichiers
input_texts = []
labels = []

for f, label in enumerate(input_files):
    with open(label, 'r') as file:
        for line in file:
            # Prétraitement de la ligne
            line = line.lower().rstrip().translate(str.maketrans('', '', ',.;:!?'))
            input_texts.append(line)
            labels.append(label)

# Étape 3: Division de corpus
train_text, test_text, Ytrain, Ytest = train_test_split(input_texts, labels, test_size=0.3)

# Étape 4: Mapping
word2idx = {'<unk>': 0}
idx = 1

for text in train_text:
    for token in text.split():
        if token not in word2idx:
            word2idx[token] = idx
            idx += 1

# Affichage du vocabulaire
print(word2idx)
print("Taille du vocabulaire:", len(word2idx))

# Étape 5: Remplacement des mots par leurs représentations numériques
def text_to_int(text):
    int_text = []
    for line in text:
        int_line = [word2idx.get(token, 0) for token in line.split()]
        int_text.append(int_line)
    return int_text

train_text_int = text_to_int(train_text)
test_text_int = text_to_int(test_text)

# Affichage pour comparer avec les données d'origine
print("Train Text (int):", train_text_int[:5])
print("Train Text:", train_text[:5])


# Étape 2.1: Création des représentations matricielles
V = len(word2idx)
A0 = np.ones((V, V))
A1 = np.ones((V, V))
pi0 = np.ones(V)
pi1 = np.ones(V)


# Étape 2.2: Extraction de caractéristiques stochastiques
def compute_counts(text_as_int, A, pi):
    for tokens in text_as_int:
        last_idx = None  # Initialisation de last_idx
        First = True
        for idx in tokens:
            if First:
                pi[idx] += 1
                First = False
            else:
                if last_idx is not None:  # Vérification si last_idx est défini
                    A[last_idx, idx] += 1
            last_idx = idx


# Exemple d'utilisation de la fonction compute_counts
compute_counts([t for t, y in zip(train_text_int, Ytrain) if y == "edgar_allan_poe.txt"], A0, pi0)
compute_counts([t for t, y in zip(train_text_int, Ytrain) if y == "robert_frost.txt"], A1, pi1)



# Étape 2.3: Normalisation
def normalize(matrix, vector):
    for i in range(matrix.shape[0]):
        row_sum = sum(matrix[i])
        if row_sum != 0:
            matrix[i] /= row_sum
    vector_sum = sum(vector)
    if vector_sum != 0:
        vector /= vector_sum

# Normalisation des matrices A0 et A1 ainsi que des listes pi0 et pi1
normalize(A0, pi0)
normalize(A1, pi1)



# Étape 2.4: Propriété log
logA0 = np.log(A0)
logA1 = np.log(A1)
logpi0 = np.log(pi0)
logpi1 = np.log(pi1)

# Étape 2.5: Priors
count0 = sum(1 for label in Ytrain if label == "edgar_allan_poe.txt")  # Nb d’éléments annotés avec label="edgar_allan_poe.txt"
count1 = sum(1 for label in Ytrain if label == "robert_frost.txt")  # Nb d’éléments annotés avec label="robert_frost.txt"
total = len(Ytrain)
p0 = count0 / total
p1 = count1 / total
logp0 = np.log(p0)
logp1 = np.log(p1)

print("Fini exo2")

class Classifier:
    def __init__(self, logAs, logpis, logpriors):
        self.logAs = logAs
        self.logpis = logpis
        self.logpriors = logpriors

    def compute_log_likelihood(self, input_, class_):
        log_likelihood = self.logpis[class_] # Initialisation avec log(pi)
        print(log_likelihood)
        for i in range(len(input_) - 1):
            log_likelihood += self.logAs[class_][input_[i]][input_[i + 1]]
        return log_likelihood

    def predict(self, inputs):
        predictions = []
        for input_ in inputs:
            # Calculer la probabilité pour chaque classe
            class_0_likelihood = self.compute_log_likelihood(input_, 0) + self.logpriors[0]
            class_1_likelihood = self.compute_log_likelihood(input_, 1) + self.logpriors[1]
            # Prédire la classe avec la plus grande probabilité
            predicted_class = np.argmax([class_0_likelihood, class_1_likelihood])
            predictions.append(predicted_class)
        return predictions


# Création d'une instance de la classe Classifier avec les listes logAs, logpis et logpriors
classifier = Classifier([logA0, logA1], [logpi0, logpi1], [logp0, logp1])
# Prédiction des classes pour les textes test_text_int
predictions = classifier.predict(test_text_int)
print("Predictions:", predictions)
