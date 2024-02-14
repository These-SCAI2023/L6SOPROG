#IMPORTS
import glob
import re
import json 
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import DistanceMetric
import seaborn as sns
import matplotlib.pyplot as plt
import csv 
import pandas as pd

#FONCTIONS
def lire_fichier (chemin):
    with open (chemin,'r', encoding='utf-8')as f:
        texte=f.read()
    return texte 

def supprimer_ponc(texte_brut):#???donne wwwww
    ponc=r'[^\w\s]|[\t\s]'
    texte=re.sub(ponc,' ',texte_brut)
    
    return texte


def texte_a_trigram(texte):
    mots=texte.split()
    liste_trigram=[]
    for m in mots :
        i=0
        while i<len(m)-3+1:
            tri=m[i:i+3]
            liste_trigram.append(tri)
            i+=1
    return liste_trigram


def compter_trigram(dic_langues,liste_trigram):
    if langue not in dic_langues:
        dic_langues[langue]={}

    for t in liste_trigram:
        if t not in dic_langues[langue]:
            dic_langues[langue][t]=1
        else :
            dic_langues[langue][t]+=1
    return dic_langues


def dic_a_liste(dic_langues):
    for langue, subdic in dic_langues.items():
        paires=[[effectif,t] for t, effectif in subdic.items()]
        liste_tri=sorted(paires,reverse=True)
        liste_freq=[t for effectif, t in liste_tri[:11]]
        #dic_mod[langue]=liste_freq
    return liste_freq


def lire_json (chemin):
    with open (chemin,"r",encoding='utf-8')as j:
        dic_j=json.load(j)
    return dic_j


def predire_langues(liste_freq,dic_mod):
    liste_pred=[]
    for langue_ref,mod in dic_mod.items():
        com=set(liste_freq).intersection(set(mod))
        liste_pred.append([len(com),langue_ref])
    liste_pred=sorted(liste_pred,reverse=True)
    return liste_pred


def compter_pred(langue, langue_pred, dic_pred):
    if langue not in dic_pred:
        dic_pred[langue]={"vp":0,"fn":0,"fp":0}
    if langue_pred not in dic_pred:
        dic_pred[langue_pred]={"vp":0,"fn":0,"fp":0}
        
    if langue_pred==langue :
        dic_pred[langue]['vp']+=1
    else :
        dic_pred[langue]['fn']+=1
        dic_pred[langue_pred]['fp']+=1
    return dic_pred




def calculer_cos(liste_freq, mod):

    a=(' ').join(liste_freq)
    b=(' ').join(mod)
    vecteur=CountVectorizer(analyzer='word')
    matrice=vecteur.fit_transform([a,b]).toarray()
    tab_dist=sklearn.metrics.pairwise.cosine_distances(matrice)
    dist_cos=tab_dist[0][1]
    return dist_cos
 
    
                                                    
#CODES

#MODELE DE LANGUES 
corpus_appr='corpus_multi/*/appr/*'
dic_langues={}
dic_mod={}
for chemin in glob.glob(corpus_appr):
    #print (chemin)
    langue=chemin.split('/')[1]
    print (langue)
    
    
    texte_brut=lire_fichier(chemin)
    #print ('texte_brut:', texte_brut[:200],'\n')
    texte=supprimer_ponc(texte_brut)
    #print ('texte sans ponctuation :', texte[:200])

    liste_trigram=texte_a_trigram(texte)
    #print (liste_tri)
    dic_langues=compter_trigram(dic_langues,liste_trigram)
    liste_freq=dic_a_liste(dic_langues)
    dic_mod[langue]=liste_freq
print (dic_mod)


#STOKER EN JSON
outpath_mod='résultat/dic_mod.json'
with open (outpath_mod,'w',encoding='utf-8')as j:
    j.write(json.dumps(dic_mod,indent=2,ensure_ascii=False))
    
dic_mod=lire_json(outpath_mod)
#print (dic_j)





#PREDICTION DE LANGUES
langues=[]
corpus_langues='corpus_multi/*'
for chemin in glob.glob(corpus_langues):
    langue=chemin.split('/')[1]
    langues.append(langue)
langues=set(langues)    
#print (langues)


dic_test={}
dic_pred={}#{langue:[liste_pred]}
dic_proches={}
for langue in langues :
    corpus_test=f'corpus_multi/{langue}/test/*'
    for chemin in glob.glob(corpus_test):
        #print ('langue test:',langue)
        
        #trouver les trigrams les plus fréquents :
        texte_brut=lire_fichier(chemin)
        texte=supprimer_ponc(texte_brut)
        liste_trigram=texte_a_trigram(texte)
        dic_test=compter_trigram(dic_test,liste_trigram)
        #print (dic_test) 
        liste_freq=dic_a_liste(dic_test)
        #print (liste_freq)
        
        
        #predire les langues :
        liste_pred=predire_langues(liste_freq,dic_mod)
        #print (liste_pred)
        langue_pred=liste_pred[0][1]
        #print ('langue pred :',langue_pred)
    
        #compter les résultat :
        dic_pred = compter_pred(langue,langue_pred,dic_pred)
        #print ('dic_pred:',dic_pred)
        
        
        
        #calculer la distance de langue proche:
        liste_proches=[p for com, p in liste_pred if com!=0]
        #print ('langues proches:',liste_proches)
        
        if langue not in dic_proches:
            dic_proches[langue]={}
            
        for p in liste_proches:
            dic_proches[langue][p]=[]
            dist_cos=calculer_cos(liste_freq, dic_mod[p])
            dic_proches[langue][p].append(dist_cos)
            
        #print ('dic_proches:',dic_proches)
        #break#chq article
    
    #break#chq langue
#print ('dic_pred:',dic_pred)
#print ('dic_proches:',dic_proches)




#STOKER EN CSV
outpath_pred='résultat/dic_pred.csv'
df=pd.DataFrame.from_dict(dic_pred,orient='index')
df.to_csv(outpath_pred)

#STOKER EN JSON
outpath_p='résultat/dic_proches.json'
with open (outpath_p, 'w',encoding='utf-8')as j:
    j.write(json.dumps(dic_proches, indent=2, ensure_ascii=False))





outpath_p='résultat/dic_proches.json'
with open (outpath_p,'r',encoding='utf-8')as j:
    dic_proches=json.load(j)
    
#PRODUIRE LE GRAPHIQUE 
outpath_png='résultat/distances_fig.png'
plt.rc('figure',figsize=(20,18))
plt.rc('font',size=20)
for langue, proches in dic_proches.items():
    #print (proches)
    langues_proches=list(proches.keys())
    #print (list(proches.values()))
    dist=[dist for subliste in list(proches.values()) for dist in subliste]
     
    sns.scatterplot(x=dist,y=langues_proches,label=langue,s=150)

plt.legend(loc='upper left',bbox_to_anchor=(1,1))
plt.grid(True)
plt.xlabel('distances')
plt.ylabel('langues')
plt.title('distances entre langue et ses langues prédites')
plt.savefig(outpath_png,dpi=300)
plt.show()
    #break #chq langue 



