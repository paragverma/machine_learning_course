# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:, [3, 4]]
X = X.as_matrix()
X = X.astype(int)

wcss = []

for i in range(1, 11):
    kmeans = KMeans(i, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel("No of clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5, random_state=0)
y_clust_pred = kmeans.fit_predict(X)

plt.scatter(X[y_clust_pred == 0, 0],  X[y_clust_pred == 0, 1], s = 100, c = 'red', label = 'C1')
plt.scatter(X[y_clust_pred == 1, 0],  X[y_clust_pred == 1, 1], s = 100, c = 'blue', label = 'C2')
plt.scatter(X[y_clust_pred == 2, 0],  X[y_clust_pred == 2, 1], s = 100, c = 'green', label = 'C3')
plt.scatter(X[y_clust_pred == 3, 0],  X[y_clust_pred == 3, 1], s = 100, c = 'cyan', label = 'C4')
plt.scatter(X[y_clust_pred == 4, 0],  X[y_clust_pred == 4, 1], s = 100, c = 'magenta', label = 'C5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'center')
plt.show()