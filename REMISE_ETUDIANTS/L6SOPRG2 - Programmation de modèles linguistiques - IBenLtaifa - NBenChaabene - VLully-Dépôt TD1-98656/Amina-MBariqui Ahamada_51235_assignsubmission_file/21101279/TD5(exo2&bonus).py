#!/usr/bin/env python
# coding: utf-8

# In[8]:


#les imports
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.ensemble import AdaBoostClassifier


# In[10]:


#MAIN 
#exo 2
df= pd.read_csv("Ressources-20240322/spam.csv", encoding="ISO 8859 1")
#print(df)


#for col in df.columns:
 #   print(col)

#effacer les colonnes inutiles
df= df.drop("Unnamed: 2", axis=1)
df= df.drop("Unnamed: 3", axis=1)
df= df.drop("Unnamed: 4", axis=1)
#print(df)

#rename les colonnes
df.rename(columns={"v1":"labels", "v2":"texte"}, inplace=True)
#print(df)

#add a third column
bi= {"ham":0, "spam":1}
df["b_labels"]= df["labels"].map(bi)
#print(df)

#variable pour mettre les elements dans la colonne labels et b_labels
labels=df["labels"]
#print(labels)
b_labels=df["b_labels"]


#create a model from the list we created
df_train,df_test,ytrain,ytest= train_test_split(labels, b_labels,test_size=0.30,train_size=0.70)
#ytrain= [str(ytrain)]

#vectorisation
tfidf= TfidfVectorizer(decode_error="ignore") #on met le vectorizer dans la variable et on precise que s'il y a des erreur de decodage, on les ignore
Xtrain=tfidf.fit_transform(df_train) #vectorisation de df_train
#print(Xtrain)
Xtest=tfidf.transform(df_test)#vectorisation de df_test

#training and evaluate
mod=MultinomialNB()
mod.fit(Xtrain,ytrain)

precision= mod.score(Xtest,ytest)
print("precision pour NB", precision) #1.0

#avec adaboost
modADA= AdaBoostClassifier()
modADA.fit(Xtrain,ytrain)
precision2= modADA.score(Xtest,ytest)
print("precision pour ADABOOST:", precision2)#1.0



# In[36]:


#BONUS
def visualize(label):
    words=""
    for message in df[df["labels"]== label]["texte"]:
        message=message.lower()
        words+= message+ " "
    wordcloud= WordCloud(width=600, height=400, background_color="white").generate(words)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
    
visual_lab= visualize(labels)

#explication
#cette fonction permets de trouver quels mots sont les plus frequents dans les
#mails,ici nous pouvons constater que les mots "now","call","will" semble etre les plus frequents


# In[32]:


pip install wordcloud


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




