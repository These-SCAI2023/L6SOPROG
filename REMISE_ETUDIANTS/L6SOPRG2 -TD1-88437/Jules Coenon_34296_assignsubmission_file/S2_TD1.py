### EXERCICE 4 -> Fonction en n_gram ###

import re
import json 
import glob
import pprint
import sklearn
from sklearn.metrics import DistanceMetric
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
    
def prediction_n_gram(chemin_fichiers_test, n): #n est le nombre de n_gram que l'on veut pour la séparation
    
    ##permet de lire le fichier
    def lire_fichier(chemin):
        with open(chemin, encoding = "utf-8") as f:
            chaine = f.read() 
        return chaine
    
    ##permet de découper le texte selon des n_grams et non pas des mots 
    def get_grams(texte, n):
        debut = 0
        n_grams = []
        for i in range(len(texte) - n+1 ):
            n_grams.append(texte[debut:n+i])
            debut += 1  
        return n_grams
    
    ##début du programme
    
    dic_langues = {}
    liste_fichiers_appr = glob.glob ("corpus_multi_appr/*/appr/*")
    print("Nombre de fichiers d'apprentissage: %i" %len(liste_fichiers_appr))

    for chemin in liste_fichiers_appr:
        dossiers = chemin.split("/")
        langue = dossiers[1]
        chaine = lire_fichier(chemin)
        liste_n_grams = get_grams(chaine, n)
    
        #on calcule les mots les plus fréquents du texte
        if langue not in dic_langues:
            dic_langues[langue] = {} #on crée un sous-dictionnaire pour une nouvelle langue

        #on stocke les effectifs
        for n_gram in liste_n_grams: #pour chaque mot de la liste de mots présents dans le texte
            if n_gram not in dic_langues[langue]: #si le mot n'est pas dans le dic_langues
                dic_langues[langue][n_gram] = 1 #on ajoute 1 au dic_langues
            else:
                dic_langues[langue][n_gram] += 1 #on augmente la valeur de 1 
        #print(dic_langues)#A_SUPPRIMER #affiche les efffectifs de chaque mot en fonction de la langue 

        dic_modeles = {}
        for langue, dic_effectifs in dic_langues.items(): #dic_effectifs rassemble les effectifs de chaque mot pour chaque langue, il est la valeur de dic_langues[langue]
            paires = [[effectif, n_gram] for n_gram, effectif in dic_effectifs.items()] #associe à la variable paires une liste comprenant un ensemble de petites listes qui regroupent l'effectif et le mot de dic_effectifs 
            liste_tri = sorted(paires)[-10:] #10 mots les plus fréquents
            dic_modeles[langue] = [n_gram      for effectif, n_gram in liste_tri]

        with open("models.json", "w", encoding= "utf-8") as w:
            w.write(json.dumps(dic_modeles, indent = 2, ensure_ascii = False)) #par défaut c'est en True
    
    with open("models.json", "r", encoding="utf-8") as f:
        dic_modeles = json.load(f)

    liste_fichiers_test = glob.glob(chemin_fichiers_test)

    VP_total = 0
    dic_stat = {}
    dic_diag = {}
    dic_exact = {}
    cosinus_distance = []
        
    for chemin in liste_fichiers_test:
        chemin = re.sub(r"[/\\\\]","/", chemin) #pour suppriemr les séparations dans le chemin, qui peuent être "/" ou "\\" selon la façon dont on rentre le chemin
        dossier = chemin.split("/") #on remplace les separations du chemin par "/" afin de séparer le chemin correctement et en ressortir la langue
        langue = dossier[1]
        chaine = lire_fichier(chemin)
        liste_n_grams = get_grams(chaine, n)

        dic_freq_texte = {} #contient les effectifs des mots du texte 
        for n_gram in liste_n_grams:
            if n_gram not in dic_freq_texte:
                dic_freq_texte[n_gram] = 1
            else:
                dic_freq_texte[n_gram] += 1

        paires = [[effectif, n_gram]    for n_gram, effectif in dic_freq_texte.items()]
        liste_tri = sorted(paires)[-10:] #10 mots les plus fréquents
        plus_frequents = {f"{langue}":[n_gram    for effectif, n_gram in liste_tri]} #set avec les 10 mots les plus fréquents de chaque texte test
        
        

        liste_predictions = []
        for langue_ref, model in dic_modeles.items():
            
            
            
            # n_gram_communs = list(plus_frequents & set(model))
            # print(n_gram_communs)
            
            Vecteur = CountVectorizer(ngram_range=(2,3), analyzer='char')
            matrice = Vecteur.fit_transform([str(plus_frequents[langue]), str(dic_modeles[langue_ref])]).toarray()
            dist_cos_tab = sklearn.metrics.pairwise.cosine_distances(matrice) # Distance avec cosinus         
            cosinus_distance.append(dist_cos_tab[0][1])
            # NB_n_gram_communs = len(n_gram_communs)
            liste_predictions.append([cosinus_distance, langue_ref])
            # print(liste_predictions)
        langue_prediction = sorted(liste_predictions)[-1][1]
        print(langue_prediction)
        print(plus_frequents)
        
        # liste_predictions = sorted(liste_predictions, reverse = True)
        
        
            
       
        # dic_cosinus = {}
        # for langue in liste_predictions:
        #     
        #     for distance in cosinus_distance:
        #         dic_cosinus[nom_fichier] = distance
                        
        print("La langue prédite pour le fichier", dossier[-1], "est:", langue_prediction) # 

        if langue not in dic_stat:
            dic_stat[langue] = {"VP":0, "FP":0, "FN":0, "VN":0}
        if langue_prediction not in dic_stat:
            dic_stat[langue_prediction] = {"VP":0, "FP":0, "FN":0, "VN":0}
        if langue_prediction == langue:  
            VP_total += 1 #sera utilisé pour calculer l'exactitude
            dic_stat[langue]["VP"] += 1
        else:
            dic_stat[langue_prediction]["FP"] += 1
            dic_stat[langue]["FN"] += 1
            
        # rappel = (dic_stat[langue]["VP"]/(dic_stat[langue]["VP"] + dic_stat[langue]["FN"])) * 100
        # precision = (dic_stat[langue]["VP"]/(dic_stat[langue]["VP"] + dic_stat[langue_prediction]["FP"])) * 100
        # f_mesure = (2*rappel*precision)/(precision + rappel)
    
        # dic_diag[langue] = {"Rappel":rappel, "Precision":precision, "F_mesure":f_mesure}
        
    exactitude = VP_total/len(liste_fichiers_test) * 100
    dic_exact = {"Exactitude du programme":exactitude}
    print(pprint.pformat(dic_stat))
    # print(pprint.pformat(dic_diag))
    print(pprint.pformat(dic_exact))

    
   
prediction_n_gram("corpus_multi_appr/*/test/*", 3) #exemple d'utilisation

# si distance cosinus 0.5 comme seuil
# pas fini