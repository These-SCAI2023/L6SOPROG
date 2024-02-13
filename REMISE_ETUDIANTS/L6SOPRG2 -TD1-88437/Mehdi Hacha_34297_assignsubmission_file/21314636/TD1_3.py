import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


with open('data.json', 'r', encoding='utf-8') as openfile:
    models = json.load(openfile)

trigrams_strings = {}
for lang, model in models.items():
    string = ' '.join(model)
    trigrams_strings[lang] = string
 
    
vectorizer = CountVectorizer()
for lang, tg in trigrams_strings.items():        
        similarities = {}
        for lang2, tg2 in trigrams_strings.items():
            if lang != lang2:
                liste = [tg, tg2]
                trigrams_vectors = vectorizer.fit_transform(liste)
                trigrams_similarity = cosine_similarity(trigrams_vectors)
                similarities[lang2] = trigrams_similarity[0][1]
        sorted_similarities = sorted(similarities.items(), key=lambda x:x[1], reverse=True)[:5]
        x = [lang for lang, similarity in sorted_similarities if similarity != 0]
        y = [similarity for lang, similarity in sorted_similarities  if similarity != 0]
        execute =  True if len(y) != 0 else False
        if execute:
            df = pd.DataFrame({"Langues": x, "Similarites": y})
            df = df.sort_values(by="Similarites", ascending=False)
            plt.figure(figsize=(6, 8))
            sns.barplot(x="Langues", y="Similarites", data=df, palette="viridis")
            plt.ylim(0, 1)
            plt.xlabel("Langues")
            plt.ylabel("Similarités par rapport à {}".format(lang))
            plt.legend()
            plt.title("Similarités de langues par rapport à {}".format(lang))
            plt.savefig(f"{lang}.png")
            plt.show()
        