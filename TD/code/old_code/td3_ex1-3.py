#!/usr/bin/env python
# coding: utf-8

# In[60]:


#imports et fonctions
import glob, re
import matplotlib.pyplot as pyplot

def lire_fichier(chemin):
  f = open(chemin, encoding="utf-8")
  chaine = f.read()
  f.close()
  return chaine

def decouper_en_mots(chaine):
  liste_mots = chaine.lower().split()
  return liste_mots
# NB: charger les fonctions importantes en amont

DATA_PATH = "../data/*"


# In[67]:


#EXERCICE 1

dic_langues = {}
dic_langues_mots = {}

for chemin in glob.glob(DATA_PATH):
  print(chemin)

  chaine = lire_fichier(chemin)
  nom_fichier = re.split("/", chemin)[-1]#Sous windows re.split("\\\\", chemin)
  nom_langue = nom_fichier[:2]
  dic_caracteres = {} 

  for carac in chaine:
    if carac not in dic_caracteres:
      dic_caracteres[carac] = 1 
    else:
      dic_caracteres[carac] += 1
    
  for ponctuation in [" ", ",", "'","\n"]:
    del dic_caracteres[ponctuation]
    
  liste_tri = [] #on va stocker dans une liste pour trier
  for caractere, effectif in dic_caracteres.items():
    liste_tri.append([effectif, caractere])#l'effectif en premier pour le tri

  liste_tri = sorted(liste_tri)
  
  dic_langues[nom_langue] = liste_tri

  dic_langues_mots[nom_langue] = decouper_en_mots(chaine)#on s'en apsse pour le moment

liste_langues = dic_langues.keys()
print("\t".join(liste_langues))
print("-"*30)
for cpt in range (1, 10):
  l = []
  for langue in liste_langues:
    l.append(dic_langues[langue][-cpt][1])
  print("\t".join(l))


# In[72]:


#EXERCICE 2

def freq_pos(liste_mots, position):
    
    position = position - 1
    dic_char = {}

    for mot in liste_mots:
        if len(mot) > position:
            if mot[position] in dic_char:
                dic_char[mot[position]] += 1
            else:
                dic_char[mot[position]] = 1

    liste_tri = [] #on va stocker dans une liste pour trier
    for caractere, effectif in dic_char.items():
        liste_tri.append([effectif, caractere])#l'effectif en premier pour le tri

    liste_tri = sorted(liste_tri, reverse=True)                
                
    return liste_tri

def afficher_freq_n(freqs_n):
    
    for cpt in range (0, 10):
        print(freqs_n[cpt][1], freqs_n[cpt][0])


# In[79]:



freqs_fr_1 = freq_pos(dic_langues_mots["fr"], 1)
freqs_fr_2 = freq_pos(dic_langues_mots["fr"], 2)
freqs_fr_3 = freq_pos(dic_langues_mots["fr"], 3)

freqs_en_1 = freq_pos(dic_langues_mots["en"], 1)
freqs_en_2 = freq_pos(dic_langues_mots["en"], 2)
freqs_en_3 = freq_pos(dic_langues_mots["en"], 3)

afficher_freq_n(freqs_en_3)


# In[101]:


#EXERCICE 3
def get_grams(texte, n):

    debut = 0
    n_grams = list()
    
    for i in range(len(a)-n+1):
        n_grams.append(texte[debut:n+i])
        debut+=1

    return n_grams 

phrase = "une phrase longue avec du texte"
phrase_ngrams= get_grams(phrase, 2)

print(phrase_ngrams)
        

