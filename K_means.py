# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 17:40:44 2022

@author: Yuanyuan_Tang

k-means



https://www.dominodatalab.com/blog/getting-started-with-k-means-clustering-in-python

https://www.w3schools.com/python/python_ml_k-means.asp
"""



import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#matplotlib inline



blobs = pd.read_csv('kmeans_blobs.csv')
colnames = list(blobs.columns[1:-1])
print(blobs.head())


'''
----------------------------------------------
Show data in two-dimension
'''
customcmap = ListedColormap(["crimson", "mediumblue", "darkmagenta"])

fig, ax = plt.subplots(figsize=(8, 6))
plt.scatter(x=blobs['x'], y=blobs['y'], s=150,
            #c=blobs['cluster'].astype('category'), 
            cmap = customcmap)
ax.set_xlabel(r'x', fontsize=14)
ax.set_ylabel(r'y', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()



'''
----------------------------------
Import kmeans from sklearn
'''
from sklearn.cluster import KMeans


data=blobs[['x','y']].values

inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


'''
----------------------------------
Cluster data into three classes 
'''
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
inertias.append(kmeans.inertia_)
plt.scatter(data[:,0], data[:,1], c=kmeans.labels_)
plt.show()