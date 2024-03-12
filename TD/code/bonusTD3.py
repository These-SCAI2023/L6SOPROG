#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:05:39 2024

@author: ceres
"""

from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
# print(kmeans.labels_)
kmeans.labels_
kmeans.predict([[0, 0], [12, 3]]) 

print(kmeans.cluster_centers_ )


# km = KMeans()
# X = np.random.rand(100, 2)
# km.fit(X)
# print(km.labels_)