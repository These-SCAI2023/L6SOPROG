# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 08:13:22 2024

@author: Ye LIU
"""

#IMPORTS
import glob
import re

import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split


#FONCTIONS
def supprimer_ponctuation (texte_brut):
    ponc=r'[^\w\s]|[\t\n]'
    texte=re.sub(ponc,'',texte_brut)
    return texte


def lire_fichier(chemin):
    with open (chemin,'r',encoding='utf-8')as f:
        ligne_brut=f.readline()
        
        while ligne_brut:
            ligne=supprimer_ponctuation(ligne_brut)
            ligne=ligne.lower()
            ligne=ligne.rstrip()
            
            input_texte.append(ligne)
            labels.append (label)
            
            ligne_brut=f.readline()

    return input_texte, labels 


def compute_counts (train_texte_int,A,pi):
    for l in train_texte_int:
        #print (l)
        for i, idx in enumerate(l):
            if i == 0:
                if idx not in pi:
                    pi[idx]=1#compter freq de premier mot?
                else :
                    pi[idx]+=1
                   
            else :
                last_idx=l[i-1]
                A[last_idx,idx]+=1 #compte la transition
                
    return A, pi


def normaliser (A, pi):
    #A:
    for y, l in enumerate(A):
        #print (y)
        som=sum(l)
        #print (som)#2662

        for x,freq in enumerate (l) :
            p = freq /som
            A[y,x]=p 
    #pi:
    somme=sum(pi)
    #print (som)#2538
    for y, freq in enumerate(pi):
        p=freq/somme
        pi[y]=p
    
    return A,pi

    




#CODES:
    
#EXO1:
#lire fichier:
input_texte=[]
labels=[]
corpus=['edgar_allan_poe.txt','robert_frost.txt']
for label, chemin in enumerate(corpus):
    #print (label,chemin)
    input_texte, labels=lire_fichier (chemin)   
    
#print (len(input_texte))#2378
#print (len(labels))#2378
    

#division :
train_texte, test_texte, y_train, y_test=train_test_split(input_texte,labels)
#print (len(train_texte))#1783
#print (len(test_texte))#595


#mapping :
word2idx={}
idx=0
for l in train_texte:
    mots=l.split(" ")
    for m in mots :
        if m not  in word2idx:
            word2idx[m]=idx
            idx+=1
#print (word2idx)#2529


#mots 2 idx:
train_texte_int=[]
test_texte_int=[]

for l in train_texte:
    l_int=[]
    mots=l.split(' ')
    for m in mots :
        idx=word2idx[m]
        l_int.append (idx)
    train_texte_int.append (l_int)
                
#print (train_texte[1])
#print (train_texte_int[1])




#EXO2:entrainement du modèle
#comptage :

A0=np.ones ((len(word2idx),len(word2idx)))#?
pi0=np.ones(len(word2idx))#n'excède jamais len(w2i)           
A0, pi0=compute_counts([t for t, y in zip (train_texte_int, y_train)if y==0], A0, pi0)
#sélectionner les lignes_int qui sont etiquetées de 0, cad issues du poème de Poe.

A1=np.ones ((len(word2idx),len(word2idx)))#?
pi1=np.ones(len(word2idx))#n'excède jamais len(w2i)           
A1, pi1=compute_counts([t for t, y in zip (train_texte_int, y_train) if y==1], A1, pi1)

A0,pi0=normaliser (A0,pi0)
A1,pi1=normaliser (A1,pi1)



#log:
logA0=np.log(A0)
logpi0=np.log(pi0)
logA1=np.log(A1)
logpi1=np.log(pi1)


#Priors :
count0= sum(y+1 for y in y_train if y ==0)#compter le nb?
count1= sum(y for y in y_train if y ==1)
total=len (y_train)
p0=count0/total
p1=count1/total
print('p0:',p0)#p0: 0.34941110487941673
print('p1:',p1)#p1: 0.6505888951205833

logp0=np.log(p0)
logp1=np.log(p1)
print (logp0)#-1.0515060990756873
print (logp1)#-0.42987733376273013









