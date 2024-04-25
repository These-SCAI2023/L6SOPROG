#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

import pandas as pd
import numpy as np
import csv


# In[3]:


data = pd.read_csv('spambase.data').values


# In[4]:


np.random.shuffle(data)
print(data)


# In[5]:


X = [i[:-1] for i in data] #données
Y = [i[-1]for i in data]   #classes


# In[6]:


Xtrain = X[:-100]
Ytrain = Y[:-100]

Xtest =  X[-100:]
Ytest = Y[-100:]


# In[7]:


model = MultinomialNB()
model.fit(Xtrain, Ytrain)
precision = model.score(Xtest, Ytest)
print("Precision pour NB:", precision)

model2 = AdaBoostClassifier()
model2.fit(Xtrain, Ytrain)
precision2 = model2.score(Xtest, Ytest)
print("Precision pour AdaB:", precision2)


# In[ ]:





# In[54]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re

def assigner_valeur(labels):
    if labels == 'ham':
        return 0
    elif labels == 'spam':
        return 1


# In[9]:


df = pd.read_csv('spam.csv', encoding = 'ISO 8859 1')


# In[22]:


print(df)


# In[23]:


#df = df.drop(df.columns[2], axis=1)


# In[20]:


df.columns = ['labels','data']


# In[21]:


df['b_labels'] = df['labels'].apply(assigner_valeur)


# In[24]:


X = df['data']
Y = df['b_labels']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)


# In[25]:


model3 = MultinomialNB()


# In[26]:


model4 = AdaBoostClassifier()


# In[27]:


tfidf = TfidfVectorizer(decode_error ='ignore')
X_train = tfidf.fit_transform(X_train) 
X_test = tfidf.transform(X_test)
#Vectorise les datas car ce sont des valeurs non numériques, et elles ne sont pas utilisable pour l'entrainement.


# In[35]:


model3.fit(X_train, Y_train)


# In[36]:


model4.fit(X_train, Y_train)


# In[58]:


#NB
precision3test = model3.score(X_test, Y_test)
precision3train = model3.score(X_train, Y_train)

print('Precision pour NB: test:', precision3test, '\n', 'Train:',precision3train)


# In[59]:


#Ada
precision4test = model4.score(X_test, Y_test)
precision4train = model4.score(X_train, Y_train)
print('Precision pour Ada: test:', precision4test, '\n', 'Train:',precision4train)


# In[ ]:


#Les modeles de test sont très proche des modeles d'entrainement, mais ne reconnait pas les spam parfaitement. 
#Il n'est pas à 100 efficace

