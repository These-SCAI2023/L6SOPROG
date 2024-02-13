# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:03:01 2024

@author: amina
"""

#touts les import
import json
import matplotlib.pyplot as plt
import glob
import sklearn
from sklearn.metrics import DistanceMetric
from sklearn.feature_extraction.text import CountVectorizer


#toutes les fonctions
def lirefich(chemin):
    with open(chemin,encoding="utf-8") as r:
        chaine= r.read()
    return chaine

def trois_carac(liste_mot):#prends les tri-grammes d'une liste
    trois_liste=[]
    for ele in liste_mot:
        three=ele[:3]
        trois_liste.append(three)
    return trois_liste

def plus_freq(dico_effec_carac):#les 10 mot les plus frequent
    pair=[[effec,caract] for caract, effec in dico_effec_carac.items()]
    liste_trie=sorted(pair)[-10:]
    return liste_trie




#MAIN
#1. CREER UN MODELE DE LANGUE
# pathfile= glob.glob("corpus_multi/corpus_multi/*/appr/*")
pathfile= glob.glob("corpus_multi/*/appr/*")
#print(pathfile)

dicolang={}
for chemin in pathfile:
    dossier=chemin.split("/")
    #print(dossier)
    langue=dossier[1]
    #print(langue)
    if langue not in dicolang:
        dicolang[langue]={}
    chaine=lirefich(chemin) # not same indentation bc il fera cette étape seulement pour les langue qui ne sont pas dans dicolang
    mots=chaine.split()
    #print(mots)
    
#je prends les tri_grammes
    tri_grammes=trois_carac(mots)
    #print(tri_grammes)

for elt in tri_grammes: 
    if elt not in dicolang[langue]: #{langue : carac : effectif}
        dicolang[langue][elt]=1
    else:
        dicolang[langue][elt]+=1
print(dicolang)
  
    
dic_modele={}
for langue, dic_effectifs in dicolang.items():
    paire=[[effectif,carac]for carac,effectif in dic_effectifs.items()] #compréhension de liste(concaténer), "paire is like paire=[]", "for carac,effectif in ... is like for carac, effectif in dicolangue:", "[effectif,carac]" is like paires.append([effectif,carac])
    #print(paire)
    liste_tri= sorted(paire)[-10:]
    #print(liste_tri)
    dic_modele[langue]=[carac for effectif, carac in liste_tri]# les 10 tri_grammes + fréquent are ds dicolangue, puis on parcours dicolangue with "[carac for effectif...]", puis on incrémente cette info with "="
    #print(dic_modele)
with open("model2.json","w",encoding="utf-8") as w:
    w.write(json.dumps(dic_modele,indent=2,ensure_ascii=False))
    
    
    
    
#2. EVALUER LE MODELES
with open("model2.json",encoding="utf-8") as f:
    modele=json.load(f)

cpt=0 # compteur mis à zero pr compter lorsque la prediction et le resultat are same
listefich_test= glob.glob("*/*/*/test/*")
#print(listefich_test)
list_predic=[]
dicfreq_txt={}
for path in listefich_test:
    dossier2= path.split("/")
    #print(dossier2)
    langue2= dossier[1]
    #print(langue2)
    string= lirefich(path)
    mot2= string.split()
    #print(mot2[:10])
    tri_grammes2=trois_carac(mot2)
    #print(tri_grammes2)
    
for element in tri_grammes2:
    if element not in dicfreq_txt:
        dicfreq_txt[element]=1
    else:
        dicfreq_txt[element]+=1
#print("le dicooooo",dicfreq_txt)

liste_tri2= plus_freq(dicfreq_txt)
#print(liste_tri2)
plusfreq_set=set([carac2 for effectif,  carac2 in liste_tri2])
#print(plusfreq_set)

# for langue_ref, model in dic_modele.items():
#     mot_commun= set(model).intersection(plusfreq_set)
#     #print(mot_commun)
#     nbr_mot_commun=len(mot_commun)
#     list_predic.append([nbr_mot_commun,langue_ref])
#     #print(list_predic)
#     list_predic=sorted(list_predic)[-1][1]
# if list_predic == langue:
#     cpt+=1
# nbr_fichier= len(glob.glob("*/*/*/test/*"))
# #print("bonne prédiction", cpt,"sur",nbr_fichier,"fichiers")
# #print(cpt/nbr_fichier)

# #SCORE DE CONFIANCE EN VP,FN etc
# score={}
# cpt_bonne_rep=0
# if list_predic not in score:
#     score[list_predic]={"VP":0,"FP":0,"FN":0}
# if list_predic==langue:
#     cpt_bonne_rep+=1
#     score[langue]["VP"]+=1
#     score[langue]["FP"]+=1
#     score[langue]["FN"]+=1
# #print("bonne prédictions",cpt_bonne_rep,"sur",nbr_fichier,"fichier",score)

    
# #3.LES AUTRES LANGUES POSSIBLES (COMPARER LE BG ET FR)
# # for fich in glob.glob("corpus_multi/corpus_multi/corpus_fr-pt_diego/fr/test/*"):
# for fich in glob.glob("corpus_multi/corpus_fr-pt_diego/fr/test/*"):
#     #print(fich)
#     dossier3= fich.split("/")
#     #print(dossier3)
#     langue3= dossier3[1]
#     #print(langue3)
#     string1= lirefich(fich)
#     mot3= string1.split()
#     #print(mot2[:10])
#     tri_grammes3=trois_carac(mot3)
#     #print(tri_grammes3)
# vect=CountVectorizer(ngram_range=(2,3), analyzer="char")
# Z=vect.fit_transform([str(tri_grammes3),str(tri_grammes2)]).toarray()
# dist_cos= sklearn.metrics.pairwise.cosine_distances(Z)
# #print(dist_cos)

# #5 GRAPHIQUE REPRESENTANT LA SIMILARITE ENTRE LES DEUX COMPARER
# plt.plot(dist_cos,marker="o")
# #plt.savefig("distance_cosFR&BG")

    
   
    


