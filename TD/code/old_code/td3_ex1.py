def lire_fichier(chemin):
  f = open(chemin)
  chaine = f.read()
  f.close()
  return chaine

def decouper_en_mots(chaine):
  liste_mots = chaine.split()
  return set(liste_mots)
# NB: charger les fonctions importantes en amont
import glob, re

dic_langues = {}
for chemin in glob.glob("../data/*"):
  print(chemin)
  chaine = lire_fichier(chemin)
  nom_fichier = re.split("/", chemin)[-1]#Sous windows re.split("\\\\", chemin)
  nom_langue = nom_fichier[:2]
  dic_caracteres = {} 
  for carac in chaine:
    if carac not in dic_caracteres:
      dic_caracteres[carac]=1 
    else:
      dic_caracteres[carac]+=1
  for ponctuation in [" ", ",", "'"]:
    del dic_caracteres[ponctuation]
  liste_tri = [] #on va stocker dans une liste pour trier
  for caractere, effectif in dic_caracteres.items():
    liste_tri.append([effectif, caractere])#l'effectif en premier pour le tri
  liste_tri = sorted(liste_tri)
#  print(liste_tri)
  dic_langues[nom_langue] = liste_tri

#  liste_mots = decouper_en_mots(chaine)#on s'en apsse pour le moment

liste_langues = dic_langues.keys()
print("\t".join(liste_langues))
print("-"*30)
for cpt in range (1, 10):
  l = []
  for langue in liste_langues:
    l.append(dic_langues[langue][-cpt][1])
  print("\t".join(l))

  
