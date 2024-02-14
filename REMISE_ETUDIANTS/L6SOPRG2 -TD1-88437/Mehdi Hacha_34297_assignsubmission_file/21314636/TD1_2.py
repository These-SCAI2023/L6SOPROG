import glob
import json


def read_file(path):
    f = open(path, encoding='utf-8')
    text = f.read()
    f.close()
    return text

def trigrams(text):   
    debut = 0
    n = 3
    trigrams_list = []
    for i in range(len(text) - 2):
        trigrams_list.append(text[debut:n])
        debut += 1
        n += 1
    return trigrams_list

def trigram_counter(trigrams_list):
    dic_counter = {}
    for tg in trigrams_list:
        if ' ' not in tg and '\n' not in tg and '-' not in tg:    
            if tg not in dic_counter:
                dic_counter[tg] = 1
            else:
                dic_counter[tg] += 1
    return dic_counter

def sorted_model(dic_counter):
    pairs = sorted([[val, k] for k, val in dic_counter.items()])[-10:]
    model = [trigram for effectif, trigram in pairs]
    return model


with open('data.json', 'r', encoding='utf-8') as openfile:
    models = json.load(openfile)

liste_langues = models.keys() 
dic_results = {lg: {"VP":0, "FP":0, "FN":0} for lg in liste_langues}
exactitude = 0
Nb_fichiers = len(glob.glob("*/*/test/*"))
for path in glob.glob('*/*/test/*'):
    language = path.split('\\')[1]
    if language not in dic_results:
        dic_results[language] = {"VP" : 0, "FP":0, "FN" :0} 
    text = read_file(path)
    trigrams_list = trigrams(text)
    dic_counter = trigram_counter(trigrams_list)
    plus_frequents = sorted_model(dic_counter)
    liste_predictions = []    
    for langue_ref, model in models.items():
        mots_communs = set(model).intersection(set(plus_frequents))     
        NB_mots_communs = len(mots_communs)
        liste_predictions.append([NB_mots_communs, langue_ref])
    liste_predictions = sorted(liste_predictions)
    lg_pred = liste_predictions[-1][1]
    if lg_pred == language:
        exactitude+=1
        dic_results[language]["VP"]+=1
    else:
        dic_results[language]["FN"]+=1
        dic_results[lg_pred]["FP"]+=1
print("Exactitude relative : ", exactitude/Nb_fichiers)
print("\trappel\t\tprécision\tf-mesure", '\n')
for langue, infos in dic_results.items():
    VP = infos["VP"]
    FP = infos["FP"]
    FN = infos["FN"]
    if VP!=0:
        rappel = VP/(VP+FN)
        precision=VP/(VP+FP)
        f_mesure = (2*rappel*precision)/(precision+rappel)
    else:
        rappel, precision, f_mesure = 0, 0, 0
    print("%s\t%2f\t%2f\t%2f"%(langue,rappel, precision, f_mesure))
