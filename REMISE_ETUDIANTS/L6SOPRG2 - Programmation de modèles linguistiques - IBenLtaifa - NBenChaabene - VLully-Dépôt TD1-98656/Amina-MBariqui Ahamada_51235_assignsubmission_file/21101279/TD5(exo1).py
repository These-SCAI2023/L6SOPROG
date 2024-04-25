#!/usr/bin/env python
# coding: utf-8

# In[6]:


#imports
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification


# In[4]:


#MAIN
#AVEC BAYES
#preparation donnees
data= pd.read_csv("Ressources-20240322/spambase.data").values
#print(len(data))
X= [i[:-1] for i in data] 
#print(len(X)) 
Y= [i[-1] for i in data]
#print(len(Y)) 

Xtrain=X[:4500]
#print(len(Xtrain))
Ytrain=Y[:4500]
#print(len(Ytrain))

Xtest= X[-100:]
print(len(Xtest))
Ytest= Y[-100:]
#print(len(Ytest))

#chargement modele
model= MultinomialNB()
model.fit(Xtrain,Ytrain)


#evaluation
precision= model.score(Xtest,Ytest)
print("precision pr NB:", precision)#0.95




# In[13]:


#AVEC ADABOOST
X,Y= make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0, random_state=0, shuffle=False)

classifier= AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=0)
classifier.fit(X,Y)

classifier.score(X,Y)#0.96


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




