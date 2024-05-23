# -*- coding: utf-8 -*-
"""
@author: Mégane
"""

#-------------------------------------Imports :--------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import glob
import random


#-----------------------Prétraitement des données textuelles :-----------------------------
data = []
labels = []

equivalences = {}
titres = []
etiquettes = []

#Pour chaque classe (auteur), pour chaque ligne du texte découpé en vers, on ajoute
# à la liste 'data' le vers, à la liste 'label' l'étiquette de l'auteur qui lui correspond (0, 1, 2 ou 3)
cpt = -1
ponct = ['.', ',', '!', '?', ':', ';', '-', '(', ')', '«', '»']
for f in glob.glob("ressources/*.txt"):
    titres.append(f.strip("ressources\"").rstrip(".txt")[1:])#la liste 'titres' ne sert qu'à la création du dictionnaire optionnel ci-dessous
    cpt += 1
    etiquettes.append(cpt)#// liste 'titres', sert au dictionnaire 'equivalences' optionnel
    for texte in open(f, 'r', encoding='utf-8') :
        txt = texte.lower().rstrip().lstrip(" ").translate(str.maketrans('', '', ''.join(ponct)))
        if len(txt) > 0 :
            data.append(txt)
            labels.append(cpt)

#Création d'un dictionnaire associant directement l'étiquette à son auteur
#par simple souci de lisibilité du résultat final
cpt = -1
for etiquette in etiquettes :    
    cpt+= 1
    equivalences[etiquette] = titres[cpt]
        
    
    
    
#-----------------------------------------Entraînement du modèle :--------------------------------------------
#Création des différentes variables employées par le modèle :
df_train, df_test, Y_train, Y_test = train_test_split(data, labels, test_size = 0.2)


tfidf = TfidfVectorizer(decode_error = "ignore")
X_train = tfidf.fit_transform(df_train)# nous changeons les valeurs de df_train pour qu'elles correspondent à notre test
X_test = tfidf.transform(df_test)# //

#entraînement du modèle :
modele = MultinomialNB()
modele.fit(X_train, Y_train)


#Génération d'un vers au hasard parmi les données de test :
ind = random.randrange(0, len(df_test) - 1, 1)


#Prédictions de l'auteur d'un vers précis tiré au hasard parmi les données de test :
pred = modele.predict(X_test)
y_pred_phrase_random = modele.predict(X_test[ind])[0]#nous cherchons la prédiction de l'auteur du vers choisi au hasard


print("-------Résultats pour un vers précis tiré au hasard dans les données de test :-------")
print("vers :", '"', df_test[ind], '"', '\n', 'auteur réel :', Y_test[ind], 'ou', equivalences[Y_test[ind]], '\n')
print("Prédiction pour le vers d'après le modèle ' :", y_pred_phrase_random, "ou", equivalences[y_pred_phrase_random], '\n', '\n')



#-------------------------------------------Résultats généraux :-----------------------------------

#Calcul de la précision des résultats :
precision = modele.score(X_test, Y_test)

#Calcul respectivement des VN(Vrais négatifs), FP (Faux positifs), FN (Faux négatifs) et TP (Vrais positifs) :
tn, fp, fn, tp = confusion_matrix(Y_test, pred, labels= [0,1,2,3])

#Création d'une matrice de confusion et de son graphique correspondant :
matr_conf = confusion_matrix(Y_test, pred)
displ = ConfusionMatrixDisplay(matr_conf).plot()

print("-------Résultats généraux :-------", '\n')
print("Précision pour MultinomialNB:", precision)
print("Vrais négatifs : ", tn, '\n', 'Faux positifs : ', fp, '\n', 'Faux négatifs : ', fn, '\n', 'Vrais positifs : ', tp, '\n')
print("précision :", precision_score(Y_test, pred, average='weighted'), '\n')
print("rappel :", recall_score(Y_test, pred, average = "weighted"), '\n')
print("F-score :", f1_score(Y_test, pred, average = "weighted"))
