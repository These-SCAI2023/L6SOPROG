# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:02:17 2024

@author: Ye LIU
"""


#IMPORTS 
import glob
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier

#CODES
#EXO1:
    
data=pd.read_csv('spambase.data').values #
#print (len(data))
np.random.shuffle(data)

#print (data)
x=[]#colonnes 
y=[]#étiquette

for l in data :
    x.append (l[:-1])
    y.append (l[-1])
# print (y)

xtrain=x[:-100]
ytrain=y[:-100]
xtest=x[-100:]
ytest=y[-100:]
#print ('train:',len(xtrain),len(ytrain))
#print ('test:',len(xtest),len(ytest))

model = MultinomialNB()
model.fit(xtrain,ytrain)
precision1=model.score(xtest,ytest)
print ('précision1 pour nb:', precision1)#0.74

model = AdaBoostClassifier()
model.fit(xtrain,ytrain)
precision2=model.score(xtest,ytest)
print ('précision2 pour nb:', precision2)#0.96





#EX02:
df=pd.read_csv('spam.csv',delimiter=',',on_bad_lines='skip', encoding='latin-1')
#print (df)#5572 lignes * 5 colonnes
#df=df.drop('v2', axis=1)#axis=1 signifie la suppression des colonnes, =0 >des lignes
#print (df)

df.rename(columns={'v1':'labels','v2':'data'},inplace=True)
#df.rename(columns={'Unnamed: 2':'b_labels'})
#print (df)



labels=df['labels'].values
#print (labels)

b_labels=[]
for l in labels:
    if l =='ham':
        b_labels.append('0')
    else :
        b_labels.append('1')
#print (b_labels)

df['b_labels']=b_labels
#print(df)



df_train, df_test=train_test_split(df,test_size=0.30)
#print (len(df_test))#1672


#print (df_test.b_labels.values)

tfidf=TfidfVectorizer(decode_error='ignore')
#initialise un tfidvectoriser qui transforme le texte en vecteurs TF-IDF
X_train=tfidf.fit_transform(df_train.data.values)
#transforme les data en vecteurs TF-IDF 
#TF:fréquences des termes dans le texte
#IDF: fréquence de document inverse, qui est une mesure d'importance d'un terme dans le corpus entier 
#print (X_train)


Y_train = df_train.b_labels.values

X_test=tfidf.transform(df_test.data.values)#idem
Y_test=df_test.b_labels.values


model = MultinomialNB()
model.fit(X_train,Y_train)
Precision1=model.score(X_test,Y_test)
print ('Précision pour nb:', Precision1)#0.958732057416268
#les deux résultat s'écartent
#parce que dans spambase.data, chaque colonne représente ses cractéristiques du texte, 
#alors qu'à partir du fichier csv, on a des texte brut et les transforme en vecteurs TF-IDF, qui prend en compte l'importance et fréquence d'un terme
#donc il serait plus précis et le modèle serait plus performant


model = AdaBoostClassifier()
model.fit(X_train,Y_train)
Precision2 =model.score(X_test,Y_test)
print ('Précision pour nb:', Precision2)#0.0.9766746411483254



















