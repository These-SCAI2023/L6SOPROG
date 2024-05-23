import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

corpus = ["Baudelaire.txt", "Verlaine.txt"]

input_texts = []
labels = []
for label, filename in enumerate(corpus):
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.rstrip().lower()
            if line:
                line = line.translate(str.maketrans('', '', string.punctuation))
                input_texts.append(line)
                labels.append(label)

train_texts, test_texts, Ytrain, Ytest = train_test_split(input_texts, labels, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()

Xtrain = vectorizer.fit_transform(train_texts)
Xtest = vectorizer.transform(test_texts)

clf = MultinomialNB()
clf.fit(Xtrain, Ytrain)

Ptrain = clf.predict(Xtrain)
train_acc = accuracy_score(Ytrain, Ptrain)
train_precision = precision_score(Ytrain, Ptrain, average='weighted')
train_recall = recall_score(Ytrain, Ptrain, average='weighted')
train_fscore = f1_score(Ytrain, Ptrain, average='weighted')
print(f"Train accuracy: {train_acc}")
print(f"Train precision: {train_precision}")
print(f"Train recall: {train_recall}")
print(f"Train F-score: {train_fscore}")

Ptest = clf.predict(Xtest)
test_acc = accuracy_score(Ytest, Ptest)
test_precision = precision_score(Ytest, Ptest, average='weighted')
test_recall = recall_score(Ytest, Ptest, average='weighted')
test_fscore = f1_score(Ytest, Ptest, average='weighted')
print(f"Test accuracy: {test_acc}")
print(f"Test precision: {test_precision}")
print(f"Test recall: {test_recall}")
print(f"Test F-score: {test_fscore}")

phrase_test = "Votre rêve familier"
phrase_test = phrase_test.lower().translate(str.maketrans('', '', string.punctuation))
phrase_test_tfidf = vectorizer.transform([phrase_test])
prediction = clf.predict(phrase_test_tfidf)

if prediction[0] == 0:
    print("La phrase semble appartenir à Baudelaire.")
else:
    print("La phrase semble appartenir à Verlaine.")
