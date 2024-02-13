
import glob
import json
liste_fichier = glob.glob("corpus_multi/*/*/*")
print("Nombre de fichier:%i"%len(liste_fichier))
#for chemin in liste_fichier:
    #print(chemin.split("/"))#pour trouver la langue

def lire_fichier(chemin):
    with open(chemin,'r',encoding = 'utf-8')as f:
        chaine = f.read()
    return chaine

dic_langues = {}#pour les différentes langues, les mots les plus fréquentes
liste_fichier_appr = glob.glob("corpus_multi/*/appr/*")
print("Nombre de Fichier d'appr:%i"%len(liste_fichier_appr))
for chemin in liste_fichier_appr:
    dossiers = chemin.split("/")
    langue = dossiers[1]
    #print(langue)
    if langue not in dic_langues:#创建sousdictionaire，用来存储最常用的词
        dic_langues[langue]={}
    chaine = lire_fichier(chemin)


#appr
dic_trigram_frequence = {}
def generate_trigram(chaine):
    chaine = ''.join(char for char in chaine if char.isalnum())
    trigrams = []
    for i in range(len(chaine)-2):
        trigram = chaine[i:i+3]
        trigrams.append(trigram)
    return trigrams
for chemin in glob.glob("corpus_multi/*/appr/*"):
    chaine = lire_fichier(chemin)
    dossiers = chemin.split("/")
    langue =  dossiers[1]

    result_trigram = generate_trigram(chaine)
    #print(result_trigram)
    if langue not in dic_trigram_frequence:
        dic_trigram_frequence[langue]={}
    
    for trigram in result_trigram:
        if trigram not in dic_trigram_frequence[langue]:
            dic_trigram_frequence[langue][trigram]=1
        else:
            dic_trigram_frequence[langue][trigram]+=1
    
#print(dic_trigram_frequence)
dic_modeles = {}
for langue, dic_effectifs in dic_trigram_frequence.items():
    paires = [[effectif,trigram]for trigram, effectif in dic_effectifs.items()]
    liste_tri = sorted(paires,reverse=True)[:10]
    dic_modeles[langue] = [trigram for _, trigram in liste_tri]  # Use _ to indicate a throwaway variable
#print(dic_modeles)




#test

import glob
liste_fichiers_test = glob.glob("corpus_multi/*/test/*")
print("Nombre de fichiers:%i"%len(liste_fichiers_test))
cpt = 0
toto = {}
dic_ngram = {}
for chemin in glob.glob("corpus_multi/*/test/*"):
    dossiers = chemin.split('/')
    langue = dossiers[1]
    chaine = lire_fichier(chemin)
    ngram_test = generate_trigram(chaine)
    if langue not in dic_ngram:
        dic_ngram[langue]={}
    for ngram in ngram_test:
        if ngram not in dic_ngram[langue]:
            dic_ngram[langue][ngram]=1
        else:
            dic_ngram[langue][ngram]+=1
    #print(dic_ngram)
    
    plus_frequents = []
    for langue,dic_effectifs in dic_ngram.items():
        paires = [[effectif,trigram]for trigram, effectif in dic_effectifs.items()]
        liste_tri = sorted(paires, reverse = True)[:10]    
        plus_frequents = [trigram for _,trigram in liste_tri]
        #print(plus_frequents)
    
    
    
#evaluation  
    
    liste_predictions = []
    for langue_ref, model in dic_modeles.items():
        mots_communs = set (plus_frequents).intersection(set(model)) #mots_communs 的集合，其中包含了 model 中的单词与 plus_frequents 集合的交集
        #print("%i mots en commun avec le modele %s" %(len(mots_communs),langue_ref))
        #print(mots_communs)
    #on va chercher automatiquement l’intersection la plus grande et v´erifiersi ¸ca correspond `a notre intuition
        
    
        NB_mots_communs = len(mots_communs)
        liste_predictions.append([len(mots_communs),langue_ref])
        #print(sorted(liste_predictions))
    langue_prediction = sorted (liste_predictions)[-1][1]
    #print(langue,langue_prediction)
    if langue_prediction not in toto:
        toto[langue_prediction]={"VP":0,"FP":0,"FN":0,"VN":0}
    if langue not in toto:
        toto[langue] = {"VP": 0, "VN": 0, "FP": 0, "FN": 0} 
    if langue_prediction == langue:
        toto[langue_prediction]["VP"]+=1
        toto[langue]["VN"]+=1
    else:
        toto[langue_prediction]["FP"]+=1
        toto[langue]["FN"]+=1
#print(toto)
    
    if langue_prediction == langue:
        cpt +=1
print("les bonnes predictions:%i"%cpt)
print(f"la percentage de bonne preduction est,{cpt/len(liste_fichiers_test)}")   

#stocker
with open('frequence.json','w',encoding = 'utf-8')as f:
    json.dump(toto,f,ensure_ascii = False, indent = 4)



#calculer la distance
import itertools

# 获取所有不同的语言组合
language_combinations = list(itertools.combinations(dic_ngram.keys(), 2))

# 计算两两语言之间的Jaccard距离
jaccard_distances = {}
for lang1, lang2 in language_combinations:
    trigrams_lang1 = set(dic_ngram[lang1].keys())
    trigrams_lang2 = set(dic_ngram[lang2].keys())

    intersection_size = len(trigrams_lang1.intersection(trigrams_lang2))
    union_size = len(trigrams_lang1.union(trigrams_lang2))
    jaccard_distance = 1 - (intersection_size / union_size)

    jaccard_distances[(lang1, lang2)] = jaccard_distance

# 打印Jaccard距离
for (lang1, lang2), distance in jaccard_distances.items():
    print(f"Jaccard distance entre ({lang1}, {lang2}): {distance}")
# %%
# %run
#dessiner
import matplotlib.pyplot as plt

# 获取所有不同的语言组合
language_combinations = list(itertools.combinations(dic_ngram.keys(), 2))

# 计算两两语言之间的Jaccard距离
jaccard_distances = {}
for lang1, lang2 in language_combinations:
    trigrams_lang1 = set(dic_ngram[lang1].keys())
    trigrams_lang2 = set(dic_ngram[lang2].keys())

    intersection_size = len(trigrams_lang1.intersection(trigrams_lang2))
    union_size = len(trigrams_lang1.union(trigrams_lang2))
    jaccard_distance = 1 - (intersection_size / union_size)

    jaccard_distances[(lang1, lang2)] = jaccard_distance

# 提取数据以便绘图
languages = list(dic_ngram.keys())
x_values = []
y_values = []

for (lang1, lang2), distance in jaccard_distances.items():
    x_values.append(distance)
    y_values.append(f'{lang1}-{lang2}')

# 绘制散点图
plt.figure(figsize=(10, 8))
plt.scatter(x_values, y_values, marker='o')

# 添加标签和标题
plt.xlabel('Jaccard Distance')
plt.ylabel('Language Pairs')
plt.title('Jaccard Distance between Language Pairs')

# 在每个点上添加语言名称
for (lang1, lang2), distance in jaccard_distances.items():
    plt.text(distance, f'{lang1}-{lang2}', f'{distance:.2f}', ha='left', va='center')

# 显示图形
plt.show()

plt.savefig("jaccard_distance.jpg")

