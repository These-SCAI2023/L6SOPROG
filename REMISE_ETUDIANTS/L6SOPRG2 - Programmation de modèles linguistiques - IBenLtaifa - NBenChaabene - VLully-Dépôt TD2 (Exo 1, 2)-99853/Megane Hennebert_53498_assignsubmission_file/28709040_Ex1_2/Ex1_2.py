# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:10:48 2024

"""
#-------------------------------Exercice 1 :---------------------------------
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split


# input_files= []
# for file in glob.glob(".txt") :
#     input_files.append(file)

input_files= []
input_files.append("edgar_allan_poe.txt")
input_files.append("robert_frost.txt")


input_texts= []
labels= []

ponct = ['.', ',', '!', '?', ':', ';', '-']

for label,f in enumerate(input_files):
    for texte in open(f, 'r', encoding= 'utf-8') :
        texte= texte.lower()
        texte= texte.rstrip()
        texte = texte.translate(str.maketrans('', '', ''.join(ponct)))
        line= texte
        input_texts.append(line)
        labels.append(label)
        
train_text, test_text, Ytrain, Ytest = train_test_split(input_texts, labels, test_size= 0.3)

# len(Ytrain), len(Ytest)
# train_text[:5]
# Ytrain[:5]

idx= 0
word2idx = {}
for ligne in input_texts :
    for word in ligne.split(' ') :
        if word not in word2idx :
            idx += 1
            word2idx[word] = idx
            
#print(word2idx)
    
tokens = []    
for ligne in train_text :
    mots = ligne.split(' ')
    for mot in mots :
        tokens.append(mot)
        # if mot not in word2idx :
        #     word2idx.append(mot)
        #     word2idx[mot]= idx
        #     idx+= 1

for token in tokens :
    if token not in word2idx :
        word2idx[token]= idx
        idx+= 1
        
#print(word2idx)
#print(len(word2idx))

train_text_int= []
test_text_int= []

for ligne in train_text :
    liste_temp= []
    mots = ligne.split(' ')
    for mot in mots :
        liste_temp.append(word2idx[mot])
    train_text_int.append(liste_temp)
        
#print(train_text_int)

for ligne in test_text :
    liste_temp= []
    mots = ligne.split(' ')
    for mot in mots :
        liste_temp.append(word2idx[mot])
    test_text_int.append(liste_temp)
    
#print(test_text_int)

print(train_text_int[50], train_text_int[100], train_text_int[150], train_text_int[200])
print("///////////////////////")
print(test_text_int[50], test_text_int[100], test_text_int[150], test_text_int[200])

#------------------------------Exercice 2 :----------------------------------

def compute_counts(text_as_int, A, pi) :
    for tokens in text_as_int :
        for ind, idx in enumerate(tokens) :
            if ind== 0 :
                pi.append(idx)
            else :
                a= tokens[ind- 1]
                b= tokens[ind - 2]
                A[b, a]+= 1

A0 = np.ones((len(word2idx), len(word2idx)))
A1= np.ones((len(word2idx), len(word2idx)))
pi0= []
pi1= []

compute_counts([t for t, y in zip(train_text_int , Ytrain)if y == 0], A0 , pi0)
compute_counts([t for t, y in zip(train_text_int , Ytrain)if y == 1], A1 , pi1)

print(A0)
print("////////////////")
print(A1)

liste0= []
for ligne in A0 :
    sum_= 0
    for chif in ligne :
        sum_+= chif
    liste0.append(sum_)
for ligne in A0 :
    for lmnt in liste0 :
        for chif in ligne :
            chif= chif/lmnt

liste1 = []
for lignee in A1 :
    sum1_= 0
    for chiff in lignee :
        sum1_+= chiff
    liste1.append(sum1_)
for lignee in A1 :
    for lmntt in liste1 :
        for chiff in lignee :
            chiff= chiff/lmntt
        
for cif0 in pi0 :
    for lmnt in liste0 :
        cif0 = cif0/lmnt
        
for cif1 in pi1 :
    for lmntt in liste1 :
        cif1 = cif1/lmntt
        
logA0 = np.log(A0)
logpi0 = np.log(pi0)

logA1 = np.log(A1)
logpi1 = np.log(pi1)

# count0 = sum(Ytrain where label==0)
# count1 = sum(Ytrain where  label== 1)

# count1 = sum(Ytrain where label== 1)
# total = len(Ytrain)
# p0 = count0/total
# p1 = count1/total

# logp0 = np.log(p0)
# logp1 = no.log(p1)

