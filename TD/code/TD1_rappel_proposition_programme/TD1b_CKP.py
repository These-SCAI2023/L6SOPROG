#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 08:43:20 2024

@author: ceres
"""

# Diagnostiquer la langue de chaque texte avec les intersections ou une distance cosinus
#Représenter graphiquement les langues les plus proches pour chaque texte

from TD1_CKP import *


path_mod="./model/"
for pth_md in glob.glob(f"{path_mod}/*10.json"):
     print(pth_md)
     model=lire_json(pth_md)


# path="./corpus_multi/*/test/*"
path="./corpus_multi/*"


for sub_path in glob.glob(path):
    print(sub_path)
    liste_vp=[]
    liste_fp=[]
    liste_vn=[]
    liste_fn=[]
    lang_txt=[]
    
    taux_taux=[] 
    # dic_perf={"VP":0,"FP":0,"VN":0,"FN":0}

    for pth_txt in glob.glob("%s/test/*"%sub_path):
        #print(pth_txt)
    
        liste_fich=[]
        liste_fich.append(pth_txt)
        langue_txt=lang(pth_txt)
        dic_lang=get_dic_langues(liste_fich)
        dic_mod=get_model(dic_lang,10)
        
        for k,v in dic_mod.items():
            lang_txt.append(pth_txt.split("/")[-1])
            langue_model=[]
            liste_taux=[]
            
            for key,val in model.items():
                langue_model.append(key)
                
                ## En utilisant une distance cosinus
                
                taux = distance_cos(str(v),str(val))*100
                
                liste_taux.append(taux)
            taux_taux.append(liste_taux)
    
    #             ## En utilisant les intersections
            
                # compare_mod=set(v).intersection(set(val))
                # taux=len(compare_mod)*10
                # liste_taux.append(taux)
                # print("Cle texte : ",k,"Cle model : ",key,"Taux : ",taux)
            #taux_taux.append(liste_taux)
            
            if taux >= 20:
                

                if k==key:
                    liste_vp.append([pth_txt,key,taux])
                    print("VP :",pth_txt, key)

                if k!=key:
                    liste_fp.append([pth_txt,key,taux])
                    print("FP",pth_txt, key)
 
            else: 
                if k==key:
                    liste_fn.append([pth_txt,key,taux])
                    print("FN",pth_txt, key)
                    
                    
                if k!=key:
                    liste_vn.append([pth_txt,key,taux])
                    print("VN",pth_txt, key)
            

##Je fabrique la matrice carrée qui comprends le nom de chaque document, le nom de chaque langue et le score 
    ar = numpy.array(taux_taux)
    
## Je fabrique le Tableau

    df = pandas.DataFrame(ar, index = lang_txt, columns = langue_model)

## Je produis la Heatmap

    fig, ax = plt.subplots(figsize=(10,20))
    hm = sns.heatmap(df)
    
    plt.savefig("%s.png"%sub_path, dpi=300, bbox_inches="tight")
    plt.clf()
           
                
                
                # Performances = VP,FP,VN,VN
                
                
               
                        
            
        
    

