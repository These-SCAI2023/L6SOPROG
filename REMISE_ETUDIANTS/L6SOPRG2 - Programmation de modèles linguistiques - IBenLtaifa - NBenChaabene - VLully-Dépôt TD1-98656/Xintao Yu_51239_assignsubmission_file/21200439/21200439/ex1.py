#!/usr/bin/env python
# coding: utf-8

# In[31]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np




# In[32]:


# Charger les données depuis le fichier CSV.
data = pd.read_csv('./spambase.data', header=None).values
# Mélanger aléatoirement les données pour garantir la randomisation.
np.random.shuffle(data)




# In[33]:


print(data)


# In[34]:


# Séparation des caractéristiques et des étiquettes.
X = data[:, :-1]
Y = data[:, -1]
# Division des données en ensembles d'entraînement et de test.
Xtrain = X[:-100] 
Ytrain = Y[:-100] 
Xtest = X[-100:]   
Ytest = Y[-100:]   



# In[37]:


# Création et entraînement du modèle Naive Bayes.
nb_model = MultinomialNB()
nb_model.fit(X_train, Y_train)

# Calcul et affichage de la précision du modèle Naive Bayes.
precision = nb_model.score(X_test , Y_test)
print(f"Précision pour NB: {precision}")

# Création et entraînement du modèle Naive Bayes.
ada_model = AdaBoostClassifier()
ada_model.fit(X_train, Y_train)

# Calcul et affichage de la précision du modèle Naive Bayes.
precision_ada = ada_model.score(X_test , Y_test)
print(f"Précision pour ada: {precision_ada}")


# In[ ]:




