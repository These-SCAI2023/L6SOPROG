#!/usr/bin/env python
# coding: utf-8

# # MODULES

# In[2]:


import glob 
import spacy
from spacy import displacy
import re
import string
import seaborn
import pandas as pd
import csv
import matplotlib.pyplot as plt
import nltk
from nltk.util import ngrams
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


# In[3]:


#!python -m spacy download en_core_web_sm
#!python.exe -m pip install --upgrade pip
#!python -m spacy download it_core_news_sm

nlp_it = spacy.load("it_core_news_sm")
nlp_fr = spacy.load("fr_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")


# # FONCTIONS

# In[4]:


def lire_fichier(chemin):
    with open(chemin, encoding ="utf-8") as f:
        chaine = f.read()
    return chaine

def tokenisation(langue):
    tok = []
    corpus = glob.glob(f"corpus_multi/{langue}/appr/*")
    nlp = eval("nlp_"+langue)
    for texte in corpus:
        text = lire_fichier(texte)
        doc = nlp(text)
        tokens = [token.text for token in doc]
        tok.extend(tokens)
    return tok


def lemmatisation(langue):
    dic_lemme = {}
    corpus = glob.glob(f"corpus_multi/{langue}/appr/*")
    nlp = eval("nlp_"+langue)
    for texte in corpus:
        text = lire_fichier(texte)
        doc = nlp(text)
        dic_freq_lemme = {}
        for token in doc:
            lemme = token.lemma_
            if lemme != "\n":
                if lemme not in dic_freq_lemme:
                    dic_freq_lemme[lemme] = 1
                else:
                    dic_freq_lemme[lemme] += 1
    return dic_freq_lemme



def REN(langue): 
    nlp = eval("nlp_"+langue)
    nlp.max_length = 2000000
    corpus = glob.glob(f"corpus_multi/{langue}/appr/*")
    for texte in corpus:
        text = lire_fichier(texte)
        doc = nlp(text)
        displacy.serve(doc, style="ent") #visualise les entités nommées
    
def REN_nettoyer(langue): #Renvoie une liste des mots de vocabulaire (hors les stop words et EN)
    nlp = eval("nlp_"+langue)
    nlp.max_length = 2000000
    corpus = glob.glob(f"corpus_multi/{langue}/appr/*")
    for texte in corpus:
        text = lire_fichier(texte)  
        for ponctuation in string.punctuation:
            text = text.replace(ponctuation,'')
        for i in ["\n", "\t", ","]:
            text =re.sub(i,"", text)
        doc = nlp(text)
        textes_nettoyer = [token.text for token in doc if not token.is_stop and not token.ent_type_]
        #displacy.serve(doc, style="ent")
    return textes_nettoyer


def len_txt(langue): #Renvoie une liste de la longueur de chaque texte brut
    lg = []
    corpus = glob.glob(f"corpus_multi/{langue}/appr/*")
    for texte in corpus:
        text = lire_fichier(texte)  
        for ponctuation in string.punctuation:
            text = text.replace(ponctuation,'')
        for i in ["\n", "\t", ","]:
            text =re.sub(i,"", text)
        lg.append(len(text.split()))
    return lg


# In[5]:


"""""
def REN_nettoyer_lg(langue): #Renvoie une liste de la longueur de chaque texte néttoyé 
    lg = []
    nlp = eval("nlp_"+langue)
    nlp.max_length = 2000000
    corpus = glob.glob(f"corpus_multi/{langue}/appr/*")
    for texte in corpus:
        text = lire_fichier(texte)  
        for ponctuation in string.punctuation:
            text = text.replace(ponctuation,'')
        for i in ["\n", "\t", ","]:
            text =re.sub(i,"", text)
        doc = nlp(text)
        textes_nettoyer = [token.text for token in doc if not token.is_stop and not token.ent_type_]
        longueur = len(textes_nettoyer)
        lg.append(longueur)
    return lg
"""""
def REN_nettoyer_lg(langue):
    vocab_txt = []
    corpus = glob.glob(f"corpus_multi/{langue}/appr/*")
    nlp = eval("nlp_"+langue)
    for texte in corpus:
        text = lire_fichier(texte)
        doc = nlp(text)
        dic_freq_lemme = {}
        for token in doc:
            lemme = token.lemma_
            if lemme != "\n":
                if lemme not in dic_freq_lemme:
                    dic_freq_lemme[lemme] = 1
                else:
                    dic_freq_lemme[lemme] += 1
        vocab_txt.append(len(dic_freq_lemme))
    return vocab_txt



def proportion_lemme(langue): 
    nb_lemme_partxt = []
    corpus = glob.glob(f"corpus_multi/{langue}/appr/*")
    nlp = eval("nlp_"+langue)
    for texte in corpus:
        lemmes = 0
        text = lire_fichier(texte)
        doc = nlp(text)
        for token in doc:
            lemma = token.lemma_
            if lemma != '\n':
                lemmes += 1
        nb_lemme_partxt.append(lemmes)#extrait les lemmes de chaque texte et les ajoute dans une liste #compte la longueur de la chaque lemme pour chaque texte #renvoie une liste de la proportion de chaque longueur de lemme par rapport au nombre total de textes
    return nb_lemme_partxt


def proportion_np(langue):
    np_par_lg = 0
    nlp = eval("nlp_"+langue)
    nlp.max_length = 2000000
    corpus = glob.glob(f"corpus_multi/{langue}/appr/*")
    for texte in corpus:
        text = lire_fichier(texte)
        for ponctuation in string.punctuation:
            text = text.replace(ponctuation,'')
        for i in ["\n", "\t", ","]:
            text =re.sub(i,"", text)
        doc = nlp(text)
        for token in doc:
            if token.ent_type_ == "PER" or "LOC":
                np_par_lg += 1
        #for l in tok:
            #longueur = len(l) #Une liste des longueurs des EN de chaque texte pour chaque langue
    #lg = round(np_par_lg/216, 2) #Renvoie une liste de la proportion de chaque longueur des EN par rapport au nombre total de textes
    return np_par_lg


# In[6]:


#!python -m spacy validate


# # MAIN

# ## Lemmatisation

# In[7]:


langues = ['fr', 'en', 'it']
dic_lemme = {langue:lemmatisation(langue) for langue in langues}
#print(dic_lemme)
for langue in langues:
    print(langue)


# ## Tokenisation

# In[8]:


Textes_EN = tokenisation("en")
Textes_FR = tokenisation("fr")
Textes_IT = tokenisation("it")


# 

# In[9]:


#REN


# In[10]:


dic_lang_len = {langue:len_txt(langue) for langue in langues}
print(dic_lang_len) #dic de toutes la longueur des textes brut


# In[11]:


dic_voc_len = {langue:REN_nettoyer_lg(langue) for langue in langues}
print(dic_voc_len) #dic du nombre de tokens (voc) par textes pour chaque langues


# In[ ]:





# In[12]:


prop_lem_general = {langue: (REN_nettoyer_lg(langue), len_txt(langue)) for langue in langues}

# Utiliser les résultats calculés
prop_lem = {}
for langue in langues:
    prop_lem[langue] = []
    for i in range(len(prop_lem_general[langue][0])):
        prop_lem[langue].append(prop_lem_general[langue][0][i] / prop_lem_general[langue][1][i]*100)
print(prop_lem)

#for i in dic_prop_lemme.values():
    #print(len(i))


# In[14]:


dic_nom_propre = {langue:proportion_np(langue) for langue in langues}
print(dic_nom_propre) #nombre de noms propres pour chaque langues
dic_np = {}
somme = sum(dic_nom_propre.values())
for cle, val in dic_nom_propre.items():
    proportion_np = round(val/somme, 6)*100
    dic_np[cle] = proportion_np
print(dic_np) #np par langue/np total


# ## GRAPHIQUES

# - dic_lang_len = Nombre de tokens par texte pour chaque langue

# In[15]:


#nombre de tokens des textes bruts
nb_textes = list(range(1, 217))

plt.subplot()
linestyles = ['-', ':', '--']
colors = ['r', 'b', 'grey']

for i, langue in enumerate(langues): #i itère chaque elt grâce à enumerate
    long_textes = dic_lang_len[langue] 
    plt.plot(nb_textes, long_textes, label=langue, color=colors[i], linestyle=linestyles[i])

plt.xlabel('Nombre de texte')
plt.ylabel('Nombre de longueurs de texte')
plt.title('Nombre de longueurs de texte par langue (brut)')

plt.legend()
plt.show()


# - dic_voc_len = Le nombre de token type (vocabulaire) par textes pour chaque langue

# In[16]:


#Nombre vocabulaire néttoyés
nb_textes = list(range(1, 217))

plt.subplot()
linestyles = ['-', ':', '--']
colors = ['r', 'b', 'grey']

for i, langue in enumerate(langues): #i itère chaque elt grâce à enumerate
    long_textes = dic_voc_len[langue] 
    plt.plot(nb_textes, long_textes, label=langue, color=colors[i], linestyle=linestyles[i])

plt.xlabel('Nombre de texte')
plt.ylabel('Nombre de vocabulaire')
plt.title('Nombre de vocabulaire par texte pour chaque langue')

plt.legend()
plt.show()


# - prop_lem = La proportion de lemmes par textes pour chaque langue

# In[17]:


#Proportion de lemmes par textes pour chaque langue
nb_textes = list(range(1, 217))

plt.subplot()
linestyles = ['-', ':', '--']
colors = ['r', 'b', 'grey']

for i, langue in enumerate(langues): #i itère chaque elt grâce à enumerate
    long_textes = prop_lem[langue] 
    plt.plot(nb_textes, long_textes, label=langue, color=colors[i], linestyle=linestyles[i])

plt.xlabel('Nombre de texte')
plt.ylabel('proportion des lemmes')
plt.title('Proportion des lemmes par textes par langue')

plt.legend()
plt.show()


# - dic_np = La proportion de noms propres pour chaque langue

# In[18]:


#proportion de noms propres pour chaque langue
nb_langues = list(range(3))

plt.subplot()
linestyles = ['-', ':', '--']
colors = ['r', 'b', 'grey']

for i, langue in enumerate(langues): #i itère chaque elt grâce à enumerate
    proportion_nom_propre = dic_np[langue] 
    plt.bar(i, proportion_nom_propre, label=langue, color=colors[i], linestyle=linestyles[i])

plt.xlabel(' ')
plt.ylabel('proportion de noms propres')
plt.title('Proportion des noms propres par langue')

plt.legend()
plt.show()


# # PARTITONNEMENT CLUSTERING 

# In[19]:


def generer_ngrams(text, n):
    characters = list(text)
    car_ngrams = list(ngrams(characters, n))
    return car_ngrams

texte = "je suis ici chez moi"

bigrammes = generer_ngrams(texte, 2)
trigrammes = generer_ngrams(texte, 3)
gram4 = generer_ngrams(texte, 4)
gram5 = generer_ngrams(texte, 5)

print(bigrammes)


# In[23]:


"""
def bigram_test(langue):
    corpus_words = []  # Liste pour stocker les mots du corpus
    corpus = glob.glob(f"corpus_multi/{langue}/appr/*")
    for texte in corpus:
        text = lire_fichier(texte)
        corpus_words.extend(text.split())  # Ajoutez les mots du texte à la liste

    bigrams = list(ngrams(corpus_words, 2))  # Générez des bigrammes à partir des mots
    bigram_strings = [' '.join(bigram) for bigram in bigrams]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(bigram_strings)

    # Perform k-means clustering
    num_clusters = 3  # You can adjust this number as needed
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)

    # Get cluster labels
    cluster_labels = kmeans.labels_

    print(X)
    print(kmeans)
    print(cluster_labels)
    # Group words based on cluster labels
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(bigram_strings[i])

    # Print clusters
    for cluster, words in clusters.items():
        print(f"Cluster {cluster}:")
        print(words)
        print()
        break

for langue in langues:
    print(bigram_test(langue))
    break
"""


# In[24]:


from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import random

def sample_clusters(clusters, sample_size=5):
    sampled_clusters = {}
    for cluster, words in clusters.items():
        if len(words) <= sample_size:
            sampled_clusters[cluster] = words
        else:
            sampled_clusters[cluster] = random.sample(words, sample_size)
    return sampled_clusters

def bigram_test(langue, sample_size=5):
    corpus_words = []  # Liste pour stocker les mots du corpus
    corpus = glob.glob(f"corpus_multi/{langue}/appr/*")
    for texte in corpus:
        text = lire_fichier(texte)
        corpus_words.extend(text.split())  # Ajoutez les mots du texte à la liste

    bigrams = list(ngrams(corpus_words, 2))  # Générez des bigrammes à partir des mots
    bigram_strings = [' '.join(bigram) for bigram in bigrams]

    vectorizer = CountVectorizer(min_df=2, max_df=0.8)
    X = vectorizer.fit_transform(bigram_strings)

    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)

    cluster_labels = kmeans.labels_

    # Group words based on cluster labels
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(bigram_strings[i])

    # Sample clusters
    sampled_clusters = sample_clusters(clusters, sample_size)

    return sampled_clusters
        
dic_clusters = {}
for langue in langues:
    dic_clusters[langue] = bigram_test(langue)
print(dic_clusters)


# In[25]:


def partition_data(langue, train_ratio=0.8):
    corpus_files = glob.glob(f"corpus_multi/{langue}/appr/*")
    random.shuffle(corpus_files)  # Mélanger aléatoirement les fichiers

    # Séparation des données en ensembles d'apprentissage et de test
    num_train = int(len(corpus_files) * train_ratio)
    train_files = corpus_files[:num_train]
    test_files = corpus_files[num_train:]

    # Lecture des fichiers d'apprentissage et de test
    X_train = [lire_fichier(file) for file in train_files]
    y_train = [langue] * len(X_train)
    X_test = [lire_fichier(file) for file in test_files]
    y_test = [langue] * len(X_test)

    return X_train, X_test, y_train, y_test


# In[26]:


from sklearn.feature_extraction.text import CountVectorizer

def extract_ngrams(X_train):
    vectorizer = CountVectorizer(ngram_range=(2, 3), analyzer='char')
    X_train_ngrams = vectorizer.fit_transform(X_train)
    return X_train_ngrams, vectorizer


# In[27]:


from sklearn.manifold import TSNE

def visualize_clusters(X, y, langue):
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='y', cmap='viridis')
    plt.colorbar(label=langue)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Visualisation des clusters de bigrammes/trigrammes')
    plt.show()


# ## Visualisation des clusters 

# In[28]:


for langue in langues:
    X_train, X_test, y_train, y_test = partition_data(langue)
    X_train_ngrams, vectorizer = extract_ngrams(X_train)
    print()
    visualize_clusters(X_train_ngrams, y_train, langue)

