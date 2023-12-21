Nous allons implanter le calcul de la distribution des mots d'un texte
(son vocabulaire) en fonction de leur taille en caractères. Ainsi, vous
devrez calculer le nombre de mots uniques pour une taille donnée comme
le montre la Figure [\[distrib\]](#distrib){reference-type="ref"
reference="distrib"}.

![Distribution des mots par rapport à leur taille en caractères dans
différentes langues[\[distrib\]]{#distrib
label="distrib"}](images/distrib.png){width=".6\textwidth"}

L'axe des abscisses représente la taille d'un mot en caractères et l'axe
des ordonnées représente le nombre de mots uniques correspondant à cette
taille.

Pour préparer votre espace de travail sur votre notebook
[Jupyter]{.smallcaps}, créez un dossier TD1 dans lequel vous
enregistrerez votre code et vos données.

Pous constituer un corpus, vous utiliserez les fichiers textes suivants
(choisissez plain text utf-8):

-   \"Le discours de la méthode\" (fr)
    <http://www.gutenberg.org/ebooks/13846>

-   \"Ulysses\" (en) <http://www.gutenberg.org/ebooks/4300>

\vspace{1cm}
**Les étapes :**

1.  **lire** les textes

2.  **découper** en mots (ou tokeniser)

3.  **compter** le nombre de mots par taille de caractères

4.  **observer** les résultats chiffrés

5.  **représenter** cela sur une courbe

**Etape 1 : lire**

On ouvre le fichier en indiquant son chemin (*path*), si vous avez bien
enregistré votre fichier au même endroit que votre code, le nom du
fichier suffit . Si ça ne marche pas c'est que tout n'est pas au bon
endroit, regardez dans l'onglet `files` de [Jupyter]{.smallcaps} pour
voir où vous êtes.

Nous allons commencer par le \"Discours de la Méthode\", si vous avez
conservé le nom d'origine il devrait s'appeler \"13846-0.txt\".

f = open(\"13846-0.txt\") chaine = f.read() f.close()

Et on affiche un bout du texte pour vérifier que ça marche :

print(chaine\[:100\])

**Etape 2 : découper**

On va très simplement découper en mots avec la **méthode** *split*

liste_mots = chaine.split()\#approximation des occurrences
print(\"Nombre de mots :

**Etape 3 : compter**

On va utiliser un **dictionnaire** (ou tableau associatif) où l'on va
stocker pour chaque longueur en caractères le nombre de mots qu'on a
rencontré. Le fonctionnement est le suivant:

-   pour chaque mot de la liste de mots, on calcule sa longueur

-   on vérifie si on a déjà rencontré un mot de cette longueur:

    -   Si c'est le premier mot pour cette longueur on crée une **clé**
        pour cette longueur à laquelle on affecte la **valeur** 1

    -   Sinon, on **incrémente** de 1 la valeur existante

dic_longueurs = \#un dictionnaire vide

for mot in liste_mots: longueur = len(mot)\#la longueur du mot if
longueur not in dic_longueurs: \#on a jamais vu cette longueur de mot
dic_longueurs\[longueur\]=1 \# else: \#on a vu cette longueur de mot
dic_longueurs\[longueur\]+=1

print(dic_longueurs)\#pour avoir une vue de ce qu'on a fait

NB: si le processus ne vous semble pas clair, ajoutez au début de la
boucle *for* deux lignes (avec l'indentation) pour suivre le processus
pas à pas :

print(dic_longueurs) dd=input(\"Appuyez sur Enter pour passer à la
suite\")

**Etape 4: observer**

Un dictionnaire n'est pas une structure de données ordonnée, pour
vérifier que'on trouve des résultats proche de l'attendu, on va afficher
le nombre d'occurences enregistré dans `dic` pour toutes les longueurs
de 1 à 30 en utilisant **l'itérateur** *range*. Dans le *print* on
utilise du **formatage de chaînes de caractères**[^1].

for toto in range(30): nbr_occurences = dic_longueurs\[toto\]
print(\"

Vous verrez que le code plante car on a des longueurs qui ne sont pas
dans le dictionnaire, on va donc améliorer le code de la façon suivante:

for toto in range(30): if toto in dic_longueurs: nbr_occurences =
dic_longueurs\[toto\] print(\" else: nbr_occurences = 0 print(\"

**Etape 5 : représenter**

Et maintenant c'est magique, on va créer une courbe grâce à la librairie
`matplotlib`. On va importer cette librairie et la renommer pour que ça
soit plus court à écrire. Puis pour avoir les valeurs à mettre sur la
courbe on va lire les valeurs dans l'ordre croissant pour les ranger
dans une liste nommée *liste*. Pyplot prend entrée un **vecteur**, une
liste de valeurs ordonnées.

import matplotlib.pyplot as pyplot \#import avec alias

liste_effectifs = \[\] for toto in range(30): if toto in
dic_longueurs:\#on a donc vu des mots de cette longueur
liste_effectifs.append(dic_longueurs\[toto\]) else:\#on en n'a pas
vu de cette longueur, on ajoute donc un 0 liste_effectifs.append(0)
pyplot.plot(liste_effectifs)\#on \"dessine\" pyplot.show()\#\"on
affiche\"

![\"Discours de la Méthode\" : nombre de mots par longueur (en
abscisse), en ordonnée
l'effectif](images/TD1_effectifs1.png){width=".5\textwidth"}

Maintenant si on veut faire le même calcul pour l'autre texte on a juste
à changer le nom du fichier dans l'étape 1 et à relancer toutes les
cellules. Mais si on avait 100 textes à faire ça ne serait pas très
pratique. Nous allons donc voir dans l'exercice suivant comment
améliorer le code.

[^1]: Voir par exemple
    <https://stackoverflow.com/questions/5082452/string-formatting-vs-format>
