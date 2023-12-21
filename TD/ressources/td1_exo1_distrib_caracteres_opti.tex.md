Pour améliorer nous allons construire des **fonctions** pour
**factoriser** les traitements et constituer une **chaîne de
traitement** fiable.

**Etape 1: lire**

Ce qui va changer ici c'est qu'on veut traiter plusieurs textes
facilement. Bien sûr on pourrait faire :

f = open(\"13846-0.txt\")\#Discours de la Methode chaine1 = f.read()
f.close() f = open(\"4300-0.txt\")\#Ulysses chaine2 = f.read() f.close()

Mais si on a 100 textes à traiter il faut 300 lignes de code, pas très
pratique.Nous allons voir avec une fonction ça marche mieux. Pour la
fabriquer il faut décomposer notre problème, voir **ce qui est
constant** (factorisable, les opérations) et **ce qui est variable**
(non factorisable, paramètre ou sortie de la fonction). On se rend vite
compte que ce que l'on veut c'est à partir d'un chemin de fichier
(**input** ou entrant) avoir son contenu sous forme de chaîne de
caractères (**output** ou sortant). Ce qui nous donne :

def lire_fichier(chemin): f = open(chemin) chaine = f.read()
f.close() return chaine

chaine1 = lire_fichier(\"13846-0.txt\")\#Discours de la Methode
chaine2 = lire_fichier(\"4300-0.txt\")\#Ulysses

**Etape 2 : découper**

Toujours réfléchir en terme d'entrant/sortant : quel est l'entrant et le
sortant qu'il faut ajouter dans le squelette ci-contre à la place des
XXX, YYY et ZZZ?

def decouper_en_mots(XXX): \#on decoupe liste_mots = YYY return
ZZZ

(à vous de réfléchir, réponse page suivante)

def decouper_en_mots(chaine): \#on decoupe liste_mots =
chaine.split() return liste_mots

liste_mots1 = decouper_en_mots(chaine1) liste_mots2 =
decouper_en_mots(chaine2)

**Etape 3 : compter**

En entrée : la liste de mots

En sortie : les effectifs

NB: vous pouvez mettre des *print* quand vous testez pour bien voir ce
qu'il se passe.

def get_effectifs(liste_mots): dic_longueurs = for mot in
liste_mots: longueur = len(mot)\#la longueur du mot if longueur not
in dic_longueurs: \#on a jamais vu cette longueur de mot
dic_longueurs\[longueur\]=1 \# else: \#on a vu cette longueur de mot
dic_longueurs\[longueur\]+=1 return dic_longueurs

Et cette fonction on va l'utiliser directemnt dans l'étape 5

**Etape 4 : observer (obsolète)**

C'était une étape de vérification devenue inutile puisqu'on n'a pas
changé les opérations effectuées.

**Etape 5 : représenter**

Ici on va pouvoir afficher les deux courbes sur la même figure et cerise
sur le gâteau on va la sauvegarder.

import matplotlib.pyplot as pyplot \#import avec alias

for liste in \[liste_mots1, liste_mots2\]: \#on a une liste de
liste pour factoriser dic_longueurs = get_effectifs(liste)
liste_effectifs = \[\] for toto in range(30): if toto in
dic_longueurs:\#on a donc vu des mots de cette longueur
liste_effectifs.append(dic_longueurs\[toto\]) else:\#on en n'a pas
vu de cette longueur, on ajoute donc un 0 liste_effectifs.append(0)
pyplot.plot(liste_effectifs)\#on \"dessine\" mais dans la boucle

pyplot.show()\#\"on affiche\" mais hors de la boucle (pour avoir tout)

![\"Discours de la Méthode\" et \"Ulysses\" : nombre de mots par
longueur (en abscisse), en ordonnée
l'effectif](images/TD1_effectifs_total.png){width=".5\textwidth"}

On se rend compte que la figure est difficile à interpréter, en effet on
travaille en valeur absolue alors que les textes sont de taille
différente. On va donc utiliser la taille de chaque texte en mots (avec
la fonction `len`) pour avoir cette fois une figure avec la proportion
de mots de chaque longueur :

\#On remplace la ligne :
liste_effectifs.append(dic_longueurs\[toto\]) \#Par:
liste_effectifs.append(dic_longueurs\[toto\]/len(liste))

![\"Discours de la Méthode\" et \"Ulysses\" : nombre de mots par
longueur (en abscisse), en ordonnée la
fréquence](images/TD1_frequences_total.png){width=".5\textwidth"}

Voir page suivante pour un bilan

\newpage
**Après une dernière étape de factorisation voici où nous en sommes :**

def lire_fichier(chemin): f = open(chemin) chaine = f.read()
f.close() return chaine

def decouper_en_mots(chaine): liste_mots = chaine.split()
return liste_mots

def get_effectifs(liste_mots): dic_longueurs = for mot in
liste_mots: longueur = len(mot) if longueur not in dic_longueurs:
dic_longueurs\[longueur\]=1 else: dic_longueurs\[longueur\]+=1
return dic_longueurs

def vecteur_longueurs(dic_longueurs): liste_effectifs = \[\]
for toto in range(30): if toto in dic_longueurs:
liste_effectifs.append(dic_longueurs\[toto\]/len(liste_mots))
else: liste_effectifs.append(0) return liste_effectifs

import matplotlib.pyplot as pyplot

for chemin in \[\"13846-0.txt\", \"4300-0.txt\"\]: chaine =
lire_fichier(chemin) liste_mots = decouper_en_mots(chaine)
dic_longueurs = get_effectifs(liste_mots) liste_effectifs =
vecteur_longueurs(dic_longueurs) pyplot.plot(liste_effectifs)

pyplot.savefig(\"frequences.png\")\#le bonus: on sauvegarde
pyplot.show()

C'est pas mal, au prochain TD on améliorera le rendu de la figure
(légende, échelle) et on travaillera sur plus de langues.
