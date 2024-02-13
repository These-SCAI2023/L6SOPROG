#bibliothèque
import glob
import json


#fonctions
def lire_fichier(chemin):
    with open(chemin, encoding="utf-8") as f:
        chaine = f.read()
    return chaine

def decouper_en_mots(chaine):
    liste_mots = chaine.split()
    return liste_mots

def trigrammes(chaine, n):
    debut = 0
    n_grams = []
    for i in range (len(chaine)-n+1):
        n_grams.append(chaine[debut:debut+n])
        debut+=1
    return n_grams

def stockage_effectif(liste_trigrammes):
    dic_freq_tri = {}
    for trig in liste_trigrammes:
        if trig not in dic_freq_tri:
            dic_freq_tri[trig] = 1
        else:
            dic_freq_tri[trig] +=1
    return dic_freq_tri
        

    
#Main
liste_fichiers = glob.glob("corpus_multi/*/*") #ouvre les listes de fichiers pour chaque langue
liste_fichiers_appr = glob.glob("corpus_multi/*/appr/*") #récupère les appr de chaque langue
dic_freq_tri = {}
dic_model = {} #dictionnaire qui va contenir les langues

    
for chemin in glob.glob("corpus_multi/*"):
    liste_trigrammes= []
    # dossiers = chemin.split("\\")
    dossiers = chemin.split("/")
    langues = dossiers[1]
    dic_model[langues] = []
    
    for chemin2 in glob.glob("corpus_multi/*/appr/*"):
        textes = lire_fichier(chemin2)
        liste_trigrammes+=trigrammes(textes, 3)
        
#Stockage effectifs des trigrammes
        dic_freq_tri = stockage_effectif(liste_trigrammes)
        liste_tri = sorted([[eff, tri] for tri, eff in dic_freq_tri.items()])[-10:]
        # dic_model[langues] = [tri for eff, tri in liste_tri]
        dic_model[langues] = [tri for eff, tri in liste_tri]

    
    
#Calcul de l'exactitude
exactitude = 0
dic_results = {}

for chemin3 in glob.glob("corpus_multi/*/test/*"):
    # dossiers2 = chemin3.split("\\")
    dossiers2 = chemin3.split("/")
    vrai_langue = dossiers2[1]
    if vrai_langue not in dic_results:
        dic_results[vrai_langue] = {"VP":0, "FP":0, "FN":0}
    txt =  lire_fichier(chemin3)
    liste_trig = trigrammes(txt,3)
    dic_eff = stockage_effectif(liste_trig)
    
    trig_freq = sorted([[eff, trig] for trig, eff in dic_eff.items()])[-10:]
    trig_juste = []
    for eff, trig in trig_freq:
        trig_juste.append(trig)
        
    dic_trig_communs = {}
    for lg, trig_model in dic_model.items():
        trig_communs = set(trig_juste).intersection(set(trig_model))
        dic_trig_communs[lg]=len(trig_communs)
    
    liste_des_pred = []
    for langue, NB_communs in dic_model.items():
        liste_des_pred.append([NB_communs, langue])
    predictions = sorted(liste_des_pred)[-1][1]
    if vrai_langue == predictions:
        exactitude+=1
        dic_results[vrai_langue]["VP"]+=1
    else:
        dic_results[vrai_langue]["FN"]+=1
        dic_results[predictions]["FP"]+=1
        
print(exactitude)

for langue, dic in dic_results.items():
    rappel = dic["VP"]/(dic["VP"]+dic["FN"])
    precision = dic["VP"]/(dic["VP"]+dic["FP"])
    F_mesure = (2*rappel*precision)/(precision+rappel)
    moyenne = (rappel+precision)/2
    print(langue, "\t",round(rappel, 3),"\t", round(precision, 3),"\t",round(moyenne,3), "\t", round(F_mesure,3))