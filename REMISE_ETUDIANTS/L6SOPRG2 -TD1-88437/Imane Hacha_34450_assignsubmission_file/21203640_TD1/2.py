import glob, json


def lire_fichier(chemin):
    with open(chemin, encoding="utf-8") as f:
        chaine = f.read()
        f.close()
        return chaine
    
def  get_trigrams(chaine):
    trigrams = [chaine[i:i+3] for i in range(len(chaine)-2)]
    return trigrams


with open("models.json", "r", encoding="UTF-8") as f:
    models=json.load(f)

VP={}
FN={}
FP={}
for langue, dic_effectifs in models.items():
    VP[langue]=0
    FN[langue]=0
    FP[langue]=0
for chemin in glob.glob("*/*/test/*"):
    dossiers = chemin.split("\\")
    langue = dossiers[1]
    dic_freq_tg ={}    
    chaine = lire_fichier(chemin)
    trigrams = get_trigrams(chaine)  
    # on stocke les effectifs
    for tg in trigrams:
        if ' ' not in tg and '\n' not in tg and '-' not in tg:   
            if tg not in dic_freq_tg:
                dic_freq_tg[tg]=1
            else:
                dic_freq_tg[tg]+=1

    paires = [[effectif,tg] for tg, effectif in dic_freq_tg.items()]
    liste_tri = sorted(paires)[-10:]
    plus_freq = set([tg for eff, tg in liste_tri ])
   
    liste_predictions=[]
    for langue_ref, model in models.items():
        mots_communs= set(model).intersection(set(plus_freq))
        NB_mots_communs=len(mots_communs)
        liste_predictions.append([NB_mots_communs, langue_ref])
    
    langue_predictions = sorted(liste_predictions)[-1][1]
    if langue_predictions==langue:
        VP[langue]+=1
    else:
        FN[langue]+=1
        FP[langue_predictions]+=1 
print("VP :", VP)
print("FN :", FN)
print("FP :", FP)