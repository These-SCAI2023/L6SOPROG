import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split
import glob
import re


input_files = []
for path in glob.glob('*.txt'):
    input_files.append(path)

def remove_punctuation(line):
    line = re.sub(r'[^\w\s]','', line)
    return line

input_texts = []
labels = []
for label, f in enumerate(input_files):
    print(label, f, end='\n')
    for line in open(f):
        line = remove_punctuation(line.rstrip().lower())
        if len(line) != 0:
            #print(line)
            input_texts.append(line)
            labels.append(label)

train_text, test_text, Ytrain, Ytest = train_test_split(input_texts, labels, test_size=0.30)


idx = 1
word2idx1 = {}
for line in train_text:
    tokens = line.split()
    for word in tokens :
        if word not in word2idx1 :
            word2idx1[word] = idx
            idx+=1

V1 = len(word2idx1)
train_text_int = []
for line in train_text:
    tokens = line.split()
    liste_tempo = []
    for word in tokens :
        liste_tempo.append(word2idx1[word])
    train_text_int.append(liste_tempo)
print(len(train_text_int))



idx = 1
word2idx2 = {}
for line in test_text:
    tokens = line.split()
    for word in tokens :
        if word not in word2idx2 :
            word2idx2[word] = idx
            idx+=1

V2 = len(word2idx2)

test_text_int = []
for line in test_text:
    tokens = line.split()
    liste_tempo = []
    for word in tokens :
        liste_tempo.append(word2idx2[word])
    test_text_int.append(liste_tempo)
print(len(test_text_int))




A0, A1 = np.ones((V1, V1)), np.ones((V2, V2))
pi0, pi1 = np.ones(V1), np.ones (V2)


def compute_counts(text_as_int, A, pi):
    for tokens in text_as_int:
        for idx, token in enumerate(tokens):
            if idx == 0:               
                pi[token] += 1
            else:               
                last_token = tokens[idx - 1]
                A[last_token, token] += 1

compute_counts([t for t, y in zip(train_text_int, Ytrain) if y == 0], A0, pi0)

def normalize_matrix_and_list(matrix, list):
    for i in range(matrix.shape[0]):
        row_sum = np.sum(matrix[i, :])
        matrix[i, :] /= row_sum
    list_sum = np.sum(list)
    list /= list_sum

normalize_matrix_and_list(A0, pi0)
normalize_matrix_and_list(A1, pi1)

logA0 = np.log(A0)
logpi0 = np.log(pi0)
logA1 = np.log(A1)
logpi1 = np.log(pi1)

class Classifier:
    def __init__(self, logAs, logpis, logpriors):
        self.logAs = logAs
        self.logpis = logpis
        self.logpriors = logpriors

    def compute_log_likelihood(self, input_, class_):
        log_likelihood = self.logpriors[class_]
        
        for idx, token in enumerate(input_):
            if idx == 0:
                log_likelihood += self.logpis[class_][token]
            else:
                last_token = input_[idx - 1]
                log_likelihood += self.logAs[class_][last_token, token]

        return log_likelihood

    def predict(self, inputs):
        predictions = []
        for input_ in inputs:
            class_probs = []
            for class_ in range(len(self.logpriors)):
                log_likelihood = self.compute_log_likelihood(input_, class_)
                class_probs.append(log_likelihood)
            predicted_class = np.argmax(class_probs)
            predictions.append(predicted_class)

        return predictions
