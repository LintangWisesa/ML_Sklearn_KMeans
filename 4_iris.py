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
model = KMeans(n_clusters = 3, random_state=0)

# training
model.fit(dfIris[['petalL', 'petalW']])

# prediction
prediksi = model.predict(dfIris[['petalL', 'petalW']])
print(prediksi)
dfIris['prediksi'] = prediksi
print(dfIris)

# split dataset: dfSetosaP, dfVersicolorP, dfVirginicaP
dfSetosaP = dfIris[dfIris['prediksi'] == 0]
dfVersicolorP = dfIris[dfIris['prediksi'] == 2]
dfVirginicaP = dfIris[dfIris['prediksi'] == 1]

# centroids
centroid = model.cluster_centers_
print(centroid)

# plot original vs k-means prediction
plt.figure('K-Means', figsize = (14,7))

# plot petal length vs petal width (original)
plt.subplot(121)
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
plt.title('Original Data')
plt.grid(True)

# plot petal length vs petal width (prediction)
plt.subplot(122)
plt.scatter(
    dfSetosaP['petalL'],
    dfSetosaP['petalW'],
    color = 'r'
)
plt.scatter(
    dfVersicolorP['petalL'],
    dfVersicolorP['petalW'],
    color = 'lightgreen'
)
plt.scatter(
    dfVirginicaP['petalL'],
    dfVirginicaP['petalW'],
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
plt.title('K-Means Prediction')
plt.grid(True)

plt.show()

