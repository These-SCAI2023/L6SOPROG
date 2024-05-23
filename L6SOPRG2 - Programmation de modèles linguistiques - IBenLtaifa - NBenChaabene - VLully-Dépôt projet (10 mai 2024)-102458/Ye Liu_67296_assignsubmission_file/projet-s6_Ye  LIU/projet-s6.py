#!/usr/bin/env python
# coding: utf-8

# In[4]:


#IMPORTS:
import glob
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score



# In[5]:


#FONCTIONS pour modèle:
def supprimer_ponctuation (ligne_brut):
    ponc=r'[^\w\s]|[\t\n\d+]'
    ligne=re.sub(ponc, '', ligne_brut)
    return ligne
#==#phrase = phrase.lower().translate(str.maketrans('', '', string.punctuation))


def lire_fichier(label, chemin):
    lignes=[]   
    labels=[]
    
    with open (chemin, 'r',encoding='utf-8')as f:
        ligne_brut=f.readline()
        
        while ligne_brut:
            ligne=supprimer_ponctuation (ligne_brut)
            #ligne=ligne_brut.translate(str.maketrans('','',string.punctuation))
            ligne=ligne.lower()
            ligne=ligne.rstrip()
            
            
            lignes.append (ligne)
            labels.append (label)
        
            ligne_brut=f.readline()           
            
    return lignes,labels


def predire_auteur(phz):    
    phz=supprimer_ponctuation(phz)
    phz=phz.lower()
    phz=phz.rstrip()
    
    X_phz=vec.transform([phz])
    pred=model.predict(X_phz)
    return pred


# In[6]:


def evaluer_modele_2(predicitions,Ytest):# 2 classes 
    #comptage:
    f={'vp':0,'vn':0,'fn':0,'fp':0} 
    for p, y in zip(predictions, Ytest):
        if p==1:#apo=0,bau=1
            if p==y:
                f['vp']+=1
            else :#p!=y
                f['fp']+=1
        else :#p==0
            if p==y:
                f['vn']+=1
            else :#p!=y
                f['fn']+=1
    #print (f) #meme fonction que cm!
    
    #calcul:
    vp=f['vp']
    fn=f['fn']
    fp=f['fp']
    rappel=vp/(vp+fn)
    precision=vp/(vp+fp)
    f_mesure=2*rappel*precision/(rappel+precision)
    
    return f, f_mesure #0.6153846153846153



def evaluer_modele_(predictions,corpus):#classes>2
    #initialiser un dictionnaire pour stocker le résultat 
    precisions={}#initialiser un dicto 
    label2auteur={}
    for label, chemin in enumerate(corpus):
        auteur = auteur=chemin.split("//")[1].split('.')[0]
        label2auteur[label]=auteur
        if auteur not in precisions :
            precisions[auteur]={'vp':0,'fn':0,'fp':0} 
    #print (label2auteur)
    #print (precisions)
    
    #compter vp, fn; fp:
    b=0
    for label, pred in zip(Ytest, predictions):
        
        label=str(label)
        pred=np.array2string(pred) #convertit le vecteur en intégral
        #print (label, pred)
        
        if label==pred:#bonne prédiction
            b+=1
            #print ('bonne prédiction')
            auteur=label2auteur[int(label)]#label a été transformé en str!        
            precisions[auteur]['vp']+=1
            
        else :#label!=pred #mauvaise prédiction 
            auteur=label2auteur[int(label)]
            auteur_pred=label2auteur[int(pred)]
            precisions[auteur]['fn']+=1
            precisions[auteur_pred]['fp']+=1
    
    #print ("nb de bonnes prédictions: ", b)
    #print (precisions)
    
    f_mesure={}
    for auteur,subdic in precisions.items():
        vp=subdic['vp']
        fn=subdic['fn']
        fp=subdic['fp']
        
        rappel=vp/(vp+fn)
        precision=vp/(vp+fp)
        f=2*rappel*precision/(rappel+precision)
            
        if auteur not in f_mesure:
            f_mesure[auteur]={}
        f_mesure[auteur]={'rappel':rappel,'précision':precision,'f-mesure':f}
    
    
    return precisions, f_mesure 


# In[8]:


#CODES :
#1.Préparer le corpus:

#corpus=['corpus_1000//apollinaire.txt', 'corpus_1000//baudelaire.txt']
corpus=['corpus_1600//apollinaire.txt', 'corpus_1600//baudelaire.txt']
#corpus=['corpus_1600//apollinaire.txt', 'corpus_1600//baudelaire.txt', 'corpus_1600//valery.txt']
#corpus=['corpus_1600//apollinaire.txt', 'corpus_1600//baudelaire.txt', 'corpus_1600//valery.txt', 'corpus_1600//prevert.txt']
#corpus=['corpus_3000//baudelaire.txt', 'corpus_3000//apollinaire.txt']


# In[9]:


#2.Prétraitement du corpus :lire+nettoyer 
corpus_poeme=[]
labels_poeme=[]
for label, chemin in enumerate(corpus):    
    #print (chemin)
    auteur=chemin.split("//")[1].split('.')[0]
    #print (f'{auteur}:{label}')
    
    lignes,labels=lire_fichier(label, chemin)    
    print (f'{auteur}:{len(lignes)}')
    
    corpus_poeme.extend(lignes)
    labels_poeme.extend(labels)
print (len(corpus_poeme))
print (len(labels_poeme))


# In[10]:


#division du corpus:    
train_text, test_text, Ytrain, Ytest=train_test_split(corpus_poeme, labels_poeme,test_size=0.30,random_state=42)
print(len(train_text))
print (len(test_text))
print (train_text[:3])
print (Ytrain[:3])


# In[11]:


#3.Convertir le texte en vecteur :
#vec=TfidfVectorizer(decode_error='ignore')
vec=CountVectorizer(decode_error='ignore')

Xtrain=vec.fit_transform(train_text)
Xtest=vec.transform(test_text)


# In[12]:


#4.Entraîner le modèle
model=MultinomialNB()#charger le modèle
#model=AdaBoostClassifier() 
model.fit(Xtrain, Ytrain)


# In[13]:


#5. Appliquer le modèle : prédictions
#demo :
#phz="Il faisait des enfants la joie et la risée."
phz='Et jamais je ne pleure et jamais je ne ris.'
pred =predire_auteur(phz)
print (pred)


# In[14]:


predictions=model.predict(Xtest)
# print (predictions)


# In[15]:


# 6. évaluer le modèle 
#1)calcule manuellement : pas satisfaisant 
#2classes:
f,f_mesure=evaluer_modele_2(predictions, Ytest)
print(f)
print ('f_mesure:',f_mesure)

#plus de 2classes :
#compter manuellement vp, fp,fn
# precisions, f_mesure_=evaluer_modele_(predictions,corpus)
# print (precisions)
# print(f_mesure_)


# In[16]:


#2) calcule automatiquement : plus juste
cm_test=confusion_matrix(Ytest, predictions)
print (cm_test)

#f1_score_=f1_score(Ytest, predictions)
#print ('f1_score_test:',f1_score_)#==f_mesure/f_mesure_
# #^seulment pour clf de 2classes

#3)calcule automatiquement
print ('report de modèle:')    
print (classification_report(Ytest,predictions))


# In[ ]:





# In[17]:


##EXTRA : 
#FONCTIONS pour le classificateur Markov
def mapping (corpus):
    idx=0
    word2idx={}
    for l in corpus:#train_text? ou tout le corpus
        for m in l.split(' '):
            if m not in word2idx:
                word2idx[m]=idx
                idx+=1
    return word2idx
    #print(word2idx)#1800 pour train_text, 2335 pour corpus



def word_to_int(texte,word2idx):
    texte_int=[]
    for l in texte:
        l_int=[]
        mots=l.split(' ')
        for m in mots :
            idx=word2idx[m]
            l_int.append (idx)
        texte_int.append (l_int)
    return texte_int


    
def compute_counts (train_text_int, A, pi):#transforme les idx en transitions
    for l in train_text_int:#train_text_int est le texte en idx, selon le word2idx
        #print (l)
        for i, idx in enumerate(l):
            if i == 0:#initial
                if idx not in pi:
                    pi[idx]=1#compter freq de premier mot?
                else :
                    pi[idx]+=1
                   
            else :
                last_idx=l[i-1]
                A[last_idx,idx]+=1 #compte la transition
                
    return A, pi


def normaliser (A, pi):#transforme la transition en pourcentage 
    #A:
    for y, l in enumerate(A):#y = axe y, l= ligne
        #print (y)
        som=sum(l)
        #print (som)#2662

        for x,freq in enumerate (l) :
            p = freq /som
            A[y,x]=p 
        #break 
            
    #pi:
    somme=sum(pi)
    #print (som)#2538
    for y, freq in enumerate(pi):
        p=freq/somme
        pi[y]=p
    
    return A,pi


def etablir_matrice(word2idx,train_text_int,Ytrain,label):
    v=len(word2idx)
    A0=np.ones((v,v))#établir une matrice, rempli par 1
    pi0=np.ones(v)
    
    A0, pi0=compute_counts([l for l, lab in zip(train_text_int,Ytrain) if lab==label], A0, pi0)
    A0, pi0=normaliser(A0, pi0)
    logA0=np.log(A0)
    logpi0=np.log(pi0)
    
    count0=[y for y in Ytrain if y==label]
    p0=len(count0)/len(Ytrain) #prior:0.43188854489164086 #pq sum(y==0)??
    logp0=np.log(p0)
    
    return logA0,logpi0,logp0



def classifier(input_int,logA,logPI,logP):
    k=len(logP)
    
    predictions=np.zeros(len(input_int))#établir une matrice pour stocker le résultat

    for i_l,l in enumerate(input_int):
             
        #calculer dans une matrice, la pb que cette ligne_int peut accumuler 
        pb=[]#pour stocker la pb obtenu sur la base de matrice diff
        for c in range (k):
            loga=logA[c]
            logpi=logPI[c]
            logp=logP[c]
            
            logpb=0
            for i, idx in enumerate(l) :#accumuler toutes les pb qu'une idx à l'autre dans cet input
                #print (i,idx)
                if i==0:#si initial
                    logpb+=logpi[idx]
                    #print (i,idx,logpi[idx])
                    #print (logpb)#
                    
                else :
                    #print(i,idx)
                    last_idx=l[i-1]
                    #print(last_idx)
                    
                    logpb+=loga[last_idx,idx]
                    #print (loga[last_idx,idx])
            #print (c,logpb)#la valeur plus petite, la pb originale plus grande
            pb.append(logpb+logp)#selon la loi naive_bayes
        
        #sélectionner la pb plus grande
        pb_max=max(pb)#?????min?
        idx_max=pb.index(pb_max)
        #print(idx_max)
        predictions[i_l]=idx_max
        
        
    return predictions


# In[18]:


#CODES : MARKOV
#1.mapping :
word2idx=mapping(corpus_poeme)
train_text_int=word_to_int(train_text, word2idx)
test_text_int=word_to_int(test_text,word2idx)


# In[19]:


#etablir les matrices :
#2 classes :
logA0,logpi0,logp0=etablir_matrice(word2idx,train_text_int,Ytrain,0)
logA1,logpi1,logp1=etablir_matrice(word2idx,train_text_int,Ytrain,1)
logA=[logA0, logA1]
logPI=[logpi0, logpi0]
logP=[logp0, logp1]


# In[ ]:


# 3 classes:
# logA0,logpi0,logp0=etablir_matrice(word2idx,train_text_int,Ytrain,0)
# logA1,logpi1,logp1=etablir_matrice(word2idx,train_text_int,Ytrain,1)
# logA2,logpi2,logp2=etablir_matrice(word2idx,train_text_int,Ytrain,2)
# logA=[logA0, logA1,logA2]
# logPI=[logpi0, logpi0,logpi2]
# logP=[logp0, logp1,logp2]


# In[21]:


#2.Appliquer le classificateur:
#demo:
#input_="L'amour s'en va comme cette eau courante"#apo=0
input_="L’Enfant déshérité s’enivre de soleil,"#bau=1
input_=supprimer_ponctuation(input_)
input_=input_.lower()
input_=input_.rstrip()
input_int=word_to_int([input_], word2idx)

input_p=classifier(input_int,logA,logPI,logP)
print ('auteur de input:',input_p)


# In[22]:


#tester le corpus :
Ptrain=classifier(train_text_int,logA,logPI,logP)
acc_train=np.mean(Ptrain==Ytrain)
#print ('train:',acc_train)

Ptest=classifier(test_text_int,logA,logPI,logP)
acc_test=np.mean(Ptest==Ytest)
#print ('test:',acc_test)


# In[23]:


#3.évaluer le modèle :
#Comptage de vp n:
cm_train=confusion_matrix(Ytrain, Ptrain)
print (cm_train)
cm_test=confusion_matrix(Ytest,Ptest)
print (cm_test)

    
#3)calcule automatiquement
print ('report de clf:')    
print (classification_report(Ytest,Ptest))


# In[ ]:





# In[ ]:




