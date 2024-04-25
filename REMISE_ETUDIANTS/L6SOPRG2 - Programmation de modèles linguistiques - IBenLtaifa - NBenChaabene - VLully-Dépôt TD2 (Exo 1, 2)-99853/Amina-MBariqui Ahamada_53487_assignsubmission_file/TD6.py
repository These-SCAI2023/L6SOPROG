#!/usr/bin/env python
# coding: utf-8

# In[1]:


#les imports
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split
import glob


# In[29]:


#les fonctions
def word_ind(liste):#une fonction qui prends pr clé un mot et valeur son indice
    ind=1
    word_id={}
    vocabulaire=[]
    for txt in liste:
        mots=txt.split()
        vocabulaire.append(mots)
    for li in vocabulaire:
        for m in li:
            if m not in word_id:
                word_id[m]=ind
                ind+=1
    return word_id

#ne fonctionne pas
def compute_counts(txt_as_int,A,pi):
    for tokens in txt_as_int:
        for index in tokens:
            if index[0]:
                 pi[index]+=1
            else:
                A[index[-1], index]+=1
            
                
    


# In[3]:


#MAIN
#EXO1

#1.2.1
input_files=["Ressources-20240329/edgar_allan_poe.txt","Ressources-20240329/robert_frost.txt"]

#1.2.2
input_txt=[]#chaque ligne de txt
labels=[]# a quel txt appartient cette ligne 0 pr edgar et 1 pr frost
for label,f in enumerate(input_files):
    #print(f,label)
    for line0 in open(f):
            #print(line0)
            line1= line0.lower().rstrip()
            #print(line)
            line=line1.translate(line1.maketrans(string.punctuation," "*len(string.punctuation)))
            #print(line)
            input_txt.append(line)
            labels.append(label)
#print(labels)

#1.3
train_text, test_text,Ytrain,Ytest= train_test_split(input_txt,labels,test_size=0.30, train_size=0.70)
#train_text a le txt pr s'entrainer, Ytrain a les etiquette qui indique a quel txt appartient la ligne           

#1.4 
word2idx_train= word_ind(train_text)
#print(word2idx_train) {token:int,token:int,token:int}

word2idx_test= word_ind(test_text)
#print(word2idx_test)

#1.5
train_text_int=[]
test_text_int=[]

for txt in train_text:
    ligne=txt.split()
    #print(ligne)
    ligne_int_train= [word2idx_train[m] for m in ligne]
    train_text_int.append(ligne_int_train)
#print(train_text_int) [[int,int,int],[int,int,int]]

for txt1 in test_text:
    ligne1=txt1.split()
    #print(ligne)
    ligne_int_test= [word2idx_test[m1] for m1 in ligne1]
    test_text_int.append(ligne_int_test)
#print(test_text_int)

    
    
    
    
        

    
    
        
        
            
    
    



    
    

      



# In[30]:


#EXO2
#print(len(word2idx_train))#2357
#print(len(word2idx_test))#1396

A0=np.ones((3753,3753))
A1=np.ones((3753,3753))

pi0=[1]
pi1=[1]

idk=compute_counts ([t for t, y in zip(train_text_int , Ytrain)if y == 0], A0 , pi0)
#print(idk)
#2.3

for x in A0:
    #print(x)
    somme=sum(x)
    #print(somme)
    divi_x= x/somme
    #print(divi_x)


    



# In[ ]:




