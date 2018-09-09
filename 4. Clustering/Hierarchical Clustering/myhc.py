# -*- coding: utf-8 -*-
import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:, [3, 4]]
X = X.as_matrix()
X = X.astype(int)

dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.show()

agc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

y_clust_pred = agc.fit_predict(X)

plt.scatter(X[y_clust_pred == 0, 0],  X[y_clust_pred == 0, 1], s = 100, c = 'red', label = 'C1')
plt.scatter(X[y_clust_pred == 1, 0],  X[y_clust_pred == 1, 1], s = 100, c = 'blue', label = 'C2')
plt.scatter(X[y_clust_pred == 2, 0],  X[y_clust_pred == 2, 1], s = 100, c = 'green', label = 'C3')
plt.scatter(X[y_clust_pred == 3, 0],  X[y_clust_pred == 3, 1], s = 100, c = 'cyan', label = 'C4')
plt.scatter(X[y_clust_pred == 4, 0],  X[y_clust_pred == 4, 1], s = 100, c = 'magenta', label = 'C5')
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'center')
plt.show()