import glob, json


def lire_fichier(chemin):
    with open(chemin, encoding="utf-8") as f:
        chaine = f.read()
        f.close()
        return chaine
    
def  get_trigrams(chaine):
    trigrams = [chaine[i:i+3] for i in range(len(chaine)-2)]
    return trigrams


dic_langues = {}
for chemin in glob.glob("*/*/appr/*"):
    dossiers = chemin.split("\\")
    langue = dossiers[1]
    #print(langue)
    # on calcule les mots les plus fréquents du texte
    if langue not in dic_langues:
    ## on crée un sous-dictionnaire pour une nouvelle langue
        dic_langues[langue] = {}
    chaine = lire_fichier(chemin)
    trigrams = get_trigrams(chaine)  
    # on stocke les effectifs
    for tg in trigrams:
        if ' ' not in tg and '\n' not in tg and '-' not in tg:   
            if tg not in dic_langues[langue]:
                dic_langues[langue][tg]=1
            else:
                dic_langues[langue][tg]+=1

dic_modeles = {}
for langue, dic_effectifs in dic_langues.items():
    paires = [[effectif,tg] for tg, effectif in dic_effectifs.items()]
    liste_tri = sorted(paires)[-10:] # les 10 mots fréquents
    dic_modeles[langue]=[tg for effectif, tg in liste_tri]

with open ("models.json","w", encoding="utf-8") as w:
    w.write(json.dumps(dic_modeles, indent =2, ensure_ascii=False))
    






