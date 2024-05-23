#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


get_ipython().system('head apollinaire.txt')
get_ipython().system('head baudelaire.txt')


# In[3]:


input_files = []
import glob
for chemin in glob.glob("*.txt"):
    filename = chemin
    input_files.append(filename)
print(input_files)


# In[4]:


input_texts = []
labels = []
for label,f in enumerate(input_files):
    print("Index:",label)
    print("Nom de fichier:",f)
    
    with open(f,"r")as file:
        for line in file:
            line = line.lower()
            line = line.rstrip()
            line = line.translate(str.maketrans('', '', string.punctuation))
            
            input_texts.append(line)
            labels.append(label)
print("input texts:",input_texts)
print("label:", labels)


# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(input_texts)
y = labels
model = MultinomialNB()
model.fit(X, y)


# In[15]:


train_text, test_text, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)
#entrainer le model
model.fit(train_text, Ytrain)
#model prediction
Ypred = model.predict(test_text)

#evaluation : report de model
print(classification_report(Ytest, Ypred))


# In[13]:


phrase = input("entrez une phrasse:")
phrase = phrase.lower().translate(str.maketrans('', '', string.punctuation))

input_vector = vectorizer.transform([phrase])
predicted_author = model.predict(input_vector)
print("l‘auteur de prediction est:", predicted_author[0])


# In[ ]:




