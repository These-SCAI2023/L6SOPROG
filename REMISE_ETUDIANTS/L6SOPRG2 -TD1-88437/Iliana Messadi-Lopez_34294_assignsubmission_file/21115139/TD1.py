# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:10:31 2024

@author: iliblock
"""

import glob
import json
import matplotlib.pyplot as plt

#%%
def lire_fich(chemin):
    with open(chemin, encoding = 'utf=8') as f:
        chaine = f.read()
    return chaine

lis_fich = glob.glob("corpus_multi/*/*/*")
for chemin in lis_fich:
    print(chemin)
    print(chemin.split("\\"))
    break


#%%

lis_fich_appr = glob.glob("corpus_multi/*/appr/*")
# for chemin in lis_fich_appr:
#     print(chemin)
#     print(chemin.split("\\"))
#     break

dic_lan = {}
tri_g = []
for chemin in lis_fich_appr:
    #print(chemin)
    dossiers = chemin.split("\\")
    #print(dossiers)
    lan = dossiers[1]
    #print(lan)
    if lan not in dic_lan:
        dic_lan[lan] = {}
    chaine = lire_fich(chemin)
    mots = chaine.split()
    for m in mots:
         #print(m[:3]) 
        for le in range(len(m)- 2):
            tri_g.append(m[le:le+3])
        
        if tri_g[-1] not in dic_lan[lan]:
            dic_lan[lan][tri_g[-1]] = 1
        else:
            dic_lan[lan][tri_g[-1]] += 1
        #print(tri_g)

#print(tri_g)
    #print(lan)
#print(dic_lan)

#%%

dic_mod = {}
for lan, dic_effec in dic_lan.items():
    pares = [[effec, mot] for mot, effec in dic_effec.items()]
    liste_tri = sorted(pares)[-10:]
    dic_mod[lan] = [mot for effectif, mot in liste_tri]
print(dic_mod)

#%%

with open("models.json", "w", encoding = 'utf-8') as w:
    w.write(json.dumps(dic_mod, indent = 2, ensure_ascii = False))
    
#%%

with open("models.json", "r", encoding = 'utf_8') as f:
    dic_mod = json.load(f)


dic_freq_trig = {}
lis_lng = []   
tri_g_t = []
nbr_bonnes_rep = 0
FP = 0
FN = 0
lis_fich_test = glob.glob("corpus_multi/*/test/*")
#print("Nombre de fichiers : %i"%len(lis_fich_test))
for chemin in lis_fich_test:
    lis_tri = []
    dossiers = chemin.split("\\")
    lan = dossiers[1]
    #lis_lang.append(lan)
    # if lan not in dic_freq_trig:
    #     dic_freq_trig[lan] = {}
    chaine = lire_fich(chemin)
    mots = chaine.split()
    for m in mots:
        for le in range(len(m)- 2):
            tri_g_t.append(m[le:le+3])
        
        if tri_g_t[-1] not in dic_freq_trig:
            dic_freq_trig[tri_g_t[-1]] = 1
        else:
            dic_freq_trig[tri_g_t[-1]] += 1
        
#print(dic_freq_trig)

#%%    
    #hasta aquí pa lo mismo de antes pero con los fich test
    pares = [[effec, mot] for mot, effec in dic_freq_trig.items()]
    #print(pares)
    lis_tri = sorted(pares)[-10:]
    print(lis_tri)
    plus_freq = set([mot for effec, mot in lis_tri])
    print(plus_freq)
    print("Document en %s" %lan)
    for lan_ref, model in dic_mod.items():
        mots_communs = set(model).intersection(plus_freq) #-----------------------aquí creas la variable con el set de trig más comunes
        print("%i tri-grammes en commun avec le modele (%s):"  %(len(mots_communs), lan_ref))
        print(mots_communs)
    
    
    lis_pred = [] #---------------------------aquí creas la lista con las predicciones de lenguas
    print("Documentsdf en %s" %lan)
    for lan_ref, model in dic_mod.items():
        mots_communs = set(model).intersection(plus_freq)
        nbr_mots_communs = len(mots_communs)
        lis_pred.append([nbr_mots_communs, lan_ref])
        #print(sorted(lis_pred))
    lan_pred = sorted(lis_pred)[-1][1]
    sorlis = sorted(lis_pred)
    
    
    if lan_pred==lan:
        nbr_bonnes_rep +=1
    elif lan_pred != lan:
        FP += 1
        for elm in sorlis:
            if elm[1] == lan:
                FN += 1
                print("Couple de langue problématique:",lan_pred,"/", lan, "Nbr de tri-grammes trouvés en commun:", elm[0])
        

    
prop_bonnes_rep = nbr_bonnes_rep/len(lis_fich_test)
print("Bonnes réponses", nbr_bonnes_rep)
NB_fichiers = len(glob.glob("corpus_multi/*/test/*"))
print("sur %i fichiers"%NB_fichiers)
print("Proportion de bonnes réponses", prop_bonnes_rep)
print("FP:",FP)
print("FN:",FN)
