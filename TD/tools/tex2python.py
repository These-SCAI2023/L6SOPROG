## Pour récupérer le code Python dans les TDs et tester

import sys, re

if len(sys.argv)!=2:
  print("Donner en argument le chemin du fichier latex dont on doit extraire le code Python")
  exit()

f = open(sys.argv[1])
lignes = f.readlines()
f.close()

isPython = False
for l in lignes:
  if "end{python}" in l:
    isPython = False
  if isPython==True:
    print(re.sub("\n", "", l))
  if "begin{python}" in l:
    isPython = True
