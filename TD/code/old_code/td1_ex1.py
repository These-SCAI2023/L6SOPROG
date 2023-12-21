###etape 1
with open("../data/13846-0.txt") as f:
  chaine = f.read()###COMMENT: 

print(chaine[:100])

###etape 2
import re
liste_mots = re.split(" ",chaine)#approximation des occurrences
print("Nombre de mots : %i"%len(liste_mots))

###etape 3
dic_longueurs = {}
for mot in liste_mots:
  longueur = len(mot)
  if longueur not in dic_longueurs:
    dic_longueurs[longueur]=1
  else:
    dic_longueurs[longueur]+=1

print(dic_longueurs)

###etape 4.1
for toto in range(30):
  if toto in dic_longueurs:
    nbr_occurences = dic_longueurs[toto]
    print("%i : %i"%(toto, nbr_occurences))
  else:
    nbr_occurences = 0 
    print("%i : %i"%(toto, nbr_occurences))


###etape 4.2
import matplotlib.pyplot as pyplot

liste_effectifs = []
for toto in range(30):
  liste_effectifs.append(dic_longueurs[toto])

pyplot.plot(liste_effectifs)
pyplot.show()


