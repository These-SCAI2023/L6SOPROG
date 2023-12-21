def lire_fichier(chemin):
  f = open(chemin)
  chaine = f.read()
  f.close()
  return chaine

def decouper_en_mots(chaine):
  liste_mots = chaine.split()
  return set(liste_mots)

def decouper_en_mots_old(chaine):
  liste_mots = chaine.split()
  return liste_mots

def get_effectifs(liste_mots):
  dic_longueurs = {}
  for mot in liste_mots:
    longueur = len(mot)
    if longueur not in dic_longueurs:
      dic_longueurs[longueur]=1 
    else:
      dic_longueurs[longueur]+=1
  return dic_longueurs

def vecteur_longueurs(dic_longueurs):
  liste_effectifs = []
  for toto in range(30):
    if toto in dic_longueurs:
      liste_effectifs.append(dic_longueurs[toto]/len(liste_mots))
    else:
      liste_effectifs.append(0)
  return liste_effectifs

import matplotlib.pyplot as pyplot

import glob, re
cpt = 0
for chemin in glob.glob("../data/*"):#à adapter
  cpt+=1
  print(chemin)
  chaine = lire_fichier(chemin)
  nom_fichier = re.split("/", chemin)[-1]#Sous windows re.split("\\\\", chemin)
  nom_pour_legende = nom_fichier[:2]
  liste_mots = decouper_en_mots(chaine)
  dic_longueurs = get_effectifs(liste_mots)
  liste_effectifs = vecteur_longueurs(dic_longueurs)
  if cpt%2==0:
    trait = "--"
  else:
    trait = "-"
  pyplot.plot(liste_effectifs, trait, label = nom_pour_legende)
  pyplot.show()

pyplot.legend(loc = "upper right")
pyplot.title("Une magnifique Courbe")
pyplot.xlabel("Longueur des Mots")
pyplot.ylabel("Fréquence")
pyplot.savefig("frequences2.png")
pyplot.show()


