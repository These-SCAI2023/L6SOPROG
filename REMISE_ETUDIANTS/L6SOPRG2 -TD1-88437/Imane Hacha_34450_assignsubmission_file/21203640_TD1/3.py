import glob
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 


with open("models.json", "r", encoding="UTF-8") as f:
    models=json.load(f)
    
trigrams_data={}
for lang, model in models.items():
    trigrams_data[lang] = ' '.join(model)
    
vectorizer = CountVectorizer()
for lang, tg in trigrams_data.items():
    similarities = {}

    for lang2, tg2 in trigrams_data.items():
        if lang != lang2:
            trigrams_vectors = vectorizer.fit_transform([tg, tg2])
            trigrams_similarity = cosine_similarity(trigrams_vectors)
            similarities[lang2] = trigrams_similarity[0][1]

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
    x = [lang for lang, similarity in sorted_similarities if similarity != 0]
    y = [similarity for lang, similarity in sorted_similarities if similarity != 0]
    execute_plot = len(y) != 0
    
    if execute_plot:
        df = pd.DataFrame({"Languages": x, "Similarities": y})
        df = df.sort_values(by="Similarities", ascending=False)

        plt.figure(figsize=(6, 8))
        sns.barplot(x="Languages", y="Similarities", data=df, palette="Set2")
        plt.ylim(0, 1)
        plt.xlabel("Languages")
        plt.ylabel("Similarities with {}".format(lang))
        plt.legend()
        plt.title("Language Similarities with {}".format(lang))
        plt.savefig(f"{lang}.png")
        plt.show()