import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
print(dir(iris))

# create dataframe
dfIris = pd.DataFrame(
    iris['data'],
    columns = ['sepalL', 'sepalW', 'petalL', 'petalW']
)
dfIris['target'] = iris['target']
dfIris['jenis'] = dfIris['target'].apply(
    lambda x: iris['target_names'][x]
)
# print(dfIris.head())

# split dataset: dfSetosa, dfVersicolor, dfVirginica
dfSetosa = dfIris[dfIris['jenis'] == 'setosa']
dfVersicolor = dfIris[dfIris['jenis'] == 'versicolor']
dfVirginica = dfIris[dfIris['jenis'] == 'virginica']

print(dfSetosa)
print(dfVersicolor)
print(dfVirginica)

# kmeans
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 3)

# training
model.fit(dfIris[['petalL', 'petalW']])

# centroids
centroid = model.cluster_centers_
print(centroid)

# plot petal length vs petal width
plt.scatter(
    dfSetosa['petalL'],
    dfSetosa['petalW'],
    color = 'r'
)
plt.scatter(
    dfVersicolor['petalL'],
    dfVersicolor['petalW'],
    color = 'lightgreen'
)
plt.scatter(
    dfVirginica['petalL'],
    dfVirginica['petalW'],
    color = 'b'
)

# plot centroids
plt.scatter(
    centroid[:,0],
    centroid[:,1],
    marker = '*',
    color = 'y',
    s = 300
)

plt.legend(['Setosa', 'Versicolor', 'Virginica', 'Centroids'])
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title('Petal length vs petal width')
plt.grid(True)
plt.show()

