# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:17:38 2024

@author: user
"""

#importation des fonctions utiles
import glob
from collections import Counter 
import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict


#importation des fonctions utiles
def lire_fichier(chemin): #pour lire des fichiers ("r") encodé en utf-8 (encodage choisi lorsque l'on télécharge pour éviter les problèmes liés aux différents caractères selon les encodages)
    with open (chemin, "r", encoding="utf-8") as fichier:
        chaine=fichier.read()
    return chaine.replace(" ", "").replace("\n", "")  # Retirer les espaces et les retours à la ligne


def generer_ngrammes(chaine, n):
    ngrammes = [chaine[i:i + n] for i in range(len(chaine) - n + 1)]
    return ngrammes

def charger_modeles(fichier):
    with open(fichier, "r", encoding="utf-8") as toto:
        dic_modeles = json.load(toto)
    return dic_modeles

chemins = glob.glob("corpus_multi/*/appr/*")


def construire_modeles(chemins, n):
    dic_langues = {}

    for chemin in chemins:
        #utilisation de os.path.normpath pour normaliser le chemin entre les systèmes d'exploitation
        chemin = os.path.normpath(chemin)
        dossiers = chemin.split(os.path.sep)  # Utiliser os.path.sep pour obtenir le séparateur de chemin approprié

        #on cherche à trouver l'index du sous-dossier "appr" dans le chemin
        index_appr = dossiers.index("appr") if "appr" in dossiers else -1

        if index_appr == -1 or index_appr == len(dossiers) - 1:
           # print(f"Chemin incorrect : {chemin}")
            continue

        langue = dossiers[index_appr - 1]  #la langue est le dossier précédent "appr"
        #print(f"Langue : {langue}")

        chaine = lire_fichier(chemin)
        ngrammes = generer_ngrammes(chaine, n)

        if langue not in dic_langues:
            dic_langues[langue] = Counter()

        dic_langues[langue].update(ngrammes)

    dic_modeles = {}
    for langue, dic_effectifs in dic_langues.items():
        paires = [[effectif, mot] for mot, effectif in dic_effectifs.items()]
        liste_tri = sorted(paires, reverse=True)[:10]
        dic_modeles[langue] = [mot for effectif, mot in liste_tri]

    return dic_modeles

#construction des modèles
dic_modeles = construire_modeles(chemins, n=3)

def sauvegarder_modeles(dic_modeles, fichier):
    with open(fichier, "w", encoding="utf-8") as w:
        w.write(json.dumps(dic_modeles, indent=2, ensure_ascii=False))

#sauvegarde des modèles dans un fichier JSON
sauvegarder_modeles(dic_modeles, "resultats_modeles.json")



def predire_langue(dic_modeles, texte, n):
    ngrammes_texte = generer_ngrammes(texte, n)
    liste_pred = []

    for langue_ref, modele in dic_modeles.items():
        ngrammes_communs = set(modele).intersection(ngrammes_texte)
        score_confiance = (len(ngrammes_communs) / len(ngrammes_texte)) * 100 if len(ngrammes_texte) > 0 else 0
        liste_pred.append([score_confiance, langue_ref])

    score_max, langue_pred = max(liste_pred, key=lambda x: x[0])
    return langue_pred, score_max

#chargement des fichiers de test
liste_fichiers_test = glob.glob("corpus_multi/*/test/*")

#chargement des modèles préalablement construits
dic_modeles = construire_modeles(glob.glob("corpus_multi/*/appr/*"), n=3)

#dictionnaire pour stocker les résultats des prédictions
resultats_predictions = {}

#prédiction et enregistrement des résultats pour chaque fichier de test
for chemin_fichier_test in liste_fichiers_test:
    nom_fichier = os.path.basename(chemin_fichier_test)
    chaine_test = lire_fichier(chemin_fichier_test)
    langue_predite, score_confiance = predire_langue(dic_modeles, chaine_test, n=3)

    #stockage du résultat dans le dictionnaire
    resultats_predictions[nom_fichier] = {"langue_predite": langue_predite, "score_confiance": score_confiance}

#on enregistre les résultats dans un fichier JSON
with open("resultats_predictions.json", "w", encoding="utf-8") as fichier_resultats:
    json.dump(resultats_predictions, fichier_resultats, indent=2, ensure_ascii=False)


#evaluation taux de réussite du programme avec les VP, FP, FN et VN
#initialisation des compteurs
VP_total = 0  # Vrais Positifs
FP_total = 0  # Faux Positifs
FN_total = 0  # Faux Négatifs
VN_total = 0  # Vrais Négatifs

#prédiction et enregistrement des résultats pour chaque fichier de test
for chemin_fichier_test in liste_fichiers_test:
    nom_fichier = os.path.basename(chemin_fichier_test)
    chaine_test = lire_fichier(chemin_fichier_test)
    langue_reelle = nom_fichier.split("_")[0]  # Extrait la langue réelle du nom du fichier
    langue_predite, score_confiance = predire_langue(dic_modeles, chaine_test, n=3)

    #on stocke le résultat dans le dictionnaire
    resultats_predictions[nom_fichier] = {"langue_reelle": langue_reelle, "langue_predite": langue_predite, "score_confiance": score_confiance}

    #évaluation de la prédiction
    if langue_predite == langue_reelle:
        VP_total += 1
    else:
        FP_total += 1

    #calcul des Faux Négatifs
    FN_total += len(dic_modeles) - 1  # Le nombre total de langues moins la langue réelle

#calcul des Vrais Négatifs
VN_total = len(liste_fichiers_test) - (VP_total + FP_total + FN_total)

#affichage des résultats
print("Vrais Positifs (VP):", VP_total)
print("Faux Positifs (FP):", FP_total)
print("Faux Négatifs (FN):", FN_total)
print("Vrais Négatifs (VN):", VN_total)
#il semblerait que mon calcul ne soit pas bon aux vues des résultats...



#on cherche les langues proches, on va les classer dans l'ordre dans un fichier JSON 
def langues_proches(dic_modeles, texte, n):
    ngrammes_texte = generer_ngrammes(texte, n)
    
    #calculer la similarité avec chaque modèle de langue
    similarites = {}
    for langue, modele in dic_modeles.items():
        ngrammes_communs = set(modele).intersection(ngrammes_texte)
        similarite = len(ngrammes_communs) / len(ngrammes_texte) if len(ngrammes_texte) > 0 else 0
        similarites[langue] = similarite
    
    #trier les langues par similarité décroissante
    langues_proches = sorted(similarites, key=similarites.get, reverse=True)
    
    return langues_proches

#chemin du dossier de test
dossier_test = "corpus_multi/*/test/*"

#dictionnaire pour stocker les résultats des langues proches
resultats_langues_proches = defaultdict(list)

#parcourir chaque dossier de langue
for dossier_langue in glob.glob("corpus_multi/*"):
    #le nom de la langue à partir du dossier
    langue = os.path.basename(dossier_langue)

    #parcourir chaque fichier de test dans le dossier de langue
    for chemin_fichier_test in glob.glob(os.path.join(dossier_langue, "test", "*")):
        #les langues proches pour le texte actuel
        langues_proches_resultat = langues_proches(dic_modeles, lire_fichier(chemin_fichier_test), n=3)

        #ajouter les langues proches à la liste du dossier
        resultats_langues_proches[langue].extend(langues_proches_resultat)

#calculer la moyenne des résultats par langue
for langue, langues_proches_dossier in resultats_langues_proches.items():
    # Comptez le nombre d'occurrences de chaque langue dans la liste
    langues_proches_compteur = Counter(langues_proches_dossier)

    #calcule de la moyenne
    nombre_fichiers = len(glob.glob(os.path.join("corpus_multi", langue, "test", "*")))
    moyenne = {langue_proche: count / nombre_fichiers for langue_proche, count in langues_proches_compteur.items()}

    #stock le résultat dans le dictionnaire
    resultats_langues_proches[langue] = moyenne

#enregistre les résultats dans un fichier JSON
with open("resultats_langues_proches.json", "w", encoding="utf-8") as fichier_resultats:
    json.dump(resultats_langues_proches, fichier_resultats, indent=2, ensure_ascii=False)
#dans le résultat, ne pas prendre en compte le chiffre à côté, uniquement le classement 

#représentation des langues les plus proches avec Matplotlib

#chargement des résultats depuis le fichier JSON
with open("resultats_langues_proches.json", "r", encoding="utf-8") as fichier_resultats:
    resultats_langues_proches = json.load(fichier_resultats)

#on parcourt chaque langue et ses résultats
for langue, moyenne_resultats in resultats_langues_proches.items():
    #trie les langues proches par moyenne décroissante
    langues_proches_classement = sorted(moyenne_resultats, key=moyenne_resultats.get, reverse=True)

    #créer un graphique en barres
    plt.figure(figsize=(10, 6))
    plt.bar(langues_proches_classement, [moyenne_resultats[langue_proche] for langue_proche in langues_proches_classement])

    #ajouter des étiquettes et un titre
    plt.xlabel("Langues Proches")
    plt.ylabel("Moyenne des Résultats")
    plt.title(f"Langues Proches de {langue}")

    #afficher le graphique
    plt.show()