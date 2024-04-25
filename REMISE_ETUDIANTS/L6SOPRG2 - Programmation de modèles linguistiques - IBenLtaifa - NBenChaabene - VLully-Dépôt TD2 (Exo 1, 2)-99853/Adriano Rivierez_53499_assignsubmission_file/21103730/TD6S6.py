#!/usr/bin/env python
# coding: utf-8

# In[105]:


import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split
import glob
import re


# In[67]:


def lire_fichier(textes):
    with open(chemin,encoding="utf-8") as f:
        chaine = f.read()
    return chaine

def remove_punct(line):
    line = re.sub(r'[^\w\s]', '', line)
    line.lower()
    line.rstrip()
    

#input_files = []


# In[39]:


for chemin in glob.glob("ressources-20240329/*"):
    nom_txt = chemin.split('\\')[1]
    #input_files.append(nom_txt)
    


# In[115]:


input_texts, labels = [], []

for a in enumerate(input_files):
    for i in a:
        label = a[0]
        f = a[1]
    print(f, label)
    textes = lire_fichier(chemin)
    
    for line in open(chemin):
        line = re.sub(r'[^\w\s]', '', line)
        line = line.lower()
        line = line.strip()
        
        if len(line) != 0:
            input_texts.append(line)
            labels.append(label)


# In[117]:


train_text, test_text, Ytrain, Ytest = train_test_split(input_texts, labels, test_size=0.30)


# In[114]:


#print(len(Ytrain), len(Ytest), train_text[:5], Ytrain[:5])


# In[ ]:


train_text_int = []
test_text_int = []


word2idx = {}
idx = 1
for line in train_text:
    line_sep = line.split()

    for token in line_sep:
        if token not in word2idx:
            word2idx[token] =idx
            idx+=1
            
for line in train_text:
    rep_numerique2 = []
    line_sep = line.split()
    for token in line_sep:
        rep_numerique.append(word2idx[token])
    train_text_int.append(rep_numerique)
#print(train_text_int)


# In[ ]:


word2idx = {}
idx = 1
for line in test_text:
    line_sep = line.split()

    for token in line_sep:
        if token not in word2idx:
            word2idx[token] =idx
            idx+=1
            
for line2 in test_text:
    rep_numerique2 = []
    line_sep2 = line2.split()
    for token in line_sep:
        rep_numerique2.append(word2idx[token])
    test_text_int.append(rep_numerique2)
print(test_text_int)


# In[ ]:




