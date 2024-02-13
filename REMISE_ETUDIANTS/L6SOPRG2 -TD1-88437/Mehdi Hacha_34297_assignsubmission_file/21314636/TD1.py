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


models = {}
for path in glob.glob('*/*/'):
    language_name = path.split('\\')[1]
    trigrams_list = []  
    for path2 in glob.glob(path + 'appr/*'):
        text = read_file(path2)
        #length = len(trigrams_list)
        trigrams_list += trigrams(text)
        #print(f'{language_name} : {len(trigrams_list) == length}')
    dic_counter = trigram_counter(trigrams_list)
    model = sorted_model(dic_counter)
    models[language_name] = model  

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(models, f, ensure_ascii=False, indent=4)