# -*- coding: utf-8 -*-
#S6TD1 :

#imports :
import json
import glob

#defs :
def lire_fichier(chemin) :
    with open(chemin, encoding= 'utf-8') as f:
        chaine= f.read()
        return chaine

def decouper_en_mots(chaine) :
    liste_mots= chaine.split()
    return liste_mots
    
def quoi_comparer(sous_liste) :
    return sous_liste[0]
    
def get_dic_langues(corpus) :
    dic_langues= {}
    liste_fichiers_appr= glob.glob(corpus)
    for chemin in liste_fichiers_appr :
        langue= chemin.split("/")[1]
        if langue not in dic_langues:
            dic_langues[langue]= {}
        mots= lire_fichier(chemin).split()
        mots= mots[:2]
        for m in mots:
            if m not in dic_langues[langue]:
                dic_langues[langue][m]= 1
            else:
                dic_langues[langue][m]+= 1         
    return dic_langues

def get_dic_modeles(dictionnaire) :
    dic_modeles= {}
    for langue, dic_effectifs in dictionnaire.items():
        paires= [[effectif, mot] for mot, effectif in dic_effectifs.items()]
        liste_tri = sorted(paires, key= quoi_comparer)[-10:]
        dic_modeles[langue]= [mot for effectif, mot in liste_tri]
    return(dic_modeles)

#traitement :
#Preparation, fichiers d'apprentissage :
dic_langues= get_dic_langues("corpus_multi/*/appr/*")

dic_modeles= get_dic_modeles(dic_langues)

la= open("dic_langues.json", "w")#on sauvegarde en dic_langues.json le dictionnaire des langues
json.dump(dic_langues, la)
la.close()

mo= open("dic_modeles.json", "w")#on sauvegarde en dic_modeles.json le dictionnaire des modèles
json.dump(dic_modeles, mo)
mo.close()

#Tests, fichiers tests :
dic_langues_tests= get_dic_langues("corpus_multi/*/test/*")

dic_modeles_tests= get_dic_modeles(dic_langues_tests)

#Predictions, premiers calculs :
with open("dic_modeles.json", "r", encoding='utf-8') as f:
    dic_modeles= json.load(f)
liste_fichiers_test= glob.glob("corpus_multi/*/test/*")
print("Nombre de fichiers : %i"%len(liste_fichiers_test))
for chemin in liste_fichiers_test :
    dossiers= chemin.split("/")
    langue= dossiers[2]
    chaine= lire_fichier(chemin)
    mots= chaine.split()
    mots= mots[:2]
    
    dic_freq_texte= {} 
    for m in mots:
        if m not in dic_freq_texte:
            dic_freq_texte[m]= 1
        else:
            dic_freq_texte[m]+= 1
    #print(dic_freq_texte)
    
    paires= [[effectif, mot] for mot, effectif in dic_freq_texte.items()]
    liste_tri= sorted(paires, key= quoi_comparer)[-10:]
    plus_frequents= set([mot for effectif, mot in liste_tri])
    #print(plus_frequents)

    print("Document en %s"%(langue))
    for langue_ref, model in dic_modeles.items():
        mots_communs= set(model).intersection(plus_frequents)
        print("%i mots en commun avec le modgle (%s):"%(len(mots_communs), langue_ref))
        #print(mots_communs)
        
    liste_predictions= []
    print("Document en %s"%langue)
    for langue_ref, model in dic_modeles.items():
        mots_communs= set(model).intersection(plus_frequents)
        NB_mots_communs= len(mots_communs)
        liste_predictions.append([NB_mots_communs, langue_ref])
        print(sorted(liste_predictions))
        
#La partie suivante présente toujours des erreurs non résolues du TD n°4 (L5SOPRG), probablement dues
#à des divisions par 0 :
#Calculs, VP, VN...:
langue_pred = "fr"
performances = {}
cpt_VP = 0
if langue_pred not in performances:
    performances[langue_pred]= {"VP":0, "FP":0, "FN":0}
if langue_pred==langue:
    cpt_VP +=1 #permet compter le nombre de bonnes prédictions
    performances[langue]["VP"]+=1
else: #langue différente de la prédiction ; on a qlqch en trop qlq part, en moins qlq part
    performances[langue_pred]["FP"]+=1
    performances[langue]["FN"]+=1

# print("Bonnes prédictions : %i"%cpt_VP)
# NB_fichiers= len(glob.glob("corpus_multi//test/"))
# print("sur %i fichiers"%NB_fichiers)
# print(performances)

# print(cpt_VP/NB_fichiers)

# for langue, perfs in performances.items(): #sinon donne seulement la clé, .items() donne le couple clé/valeur
#     print(langue)
#     print(perfs)
#     #Rappel= VP/(VP+FN), Précision= VP/(VP+FP), quand on a un on a l'autre, ce qui manque = silence, en trop = bruit
#     VP= perfs["VP"]
#     FP= perfs["FP"]
#     FN= perfs["FN"]
#     #print(VP, FP, FN)
#     rappel= round(VP/(VP+FP), 4) #round = arrondir le résultat
#     precision= VP/(VP+FN)
#     Fmesure= round(2*rappel*precision/(precision+rappel), 4)
#     print(rappel, precision, Fmesure)
