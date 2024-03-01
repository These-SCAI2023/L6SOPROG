#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:50:09 2024

@author: ceres
"""
import json
import glob
import csv

def lire_fichier (chemin):
    liste_r=[]
    with open(chemin, newline='') as csv_file: 
        spamreader =csv.reader(csv_file, delimiter=" ", quotechar="\n")
        for row in spamreader:
            liste_r.append(row)
    return liste_r

path_data="../ressources/DATA/*/*/*.bio"

for chemin in glob.glob(path_data):
    print(chemin)
    data=lire_fichier(chemin)
    