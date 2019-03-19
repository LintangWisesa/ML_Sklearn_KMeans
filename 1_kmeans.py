import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
df = pd.read_excel('data.xlsx')
# print(df.head(2))

# plot data
# plt.scatter(df['usia'], df['gaji'])
# plt.xlabel('Usia')
# plt.ylabel('Gaji')
# plt.grid(True)
# plt.show()

# kmeans
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 2)

# ======================
# # training
model.fit(df[['usia', 'gaji']])

# # predict
print(df['usia'].values)
prediksi = model.predict(df[['usia', 'gaji']])
print(prediksi)
# ======================

# =====================
# fit & predict
# prediksi = model.fit_predict(df[['usia', 'gaji']])
# =====================

# add prediction result to dataframe
df['cluster'] = prediksi
# print(df)

# split dataframe based on its cluster
df0 = df[df['cluster'] == 0]
df1 = df[df['cluster'] == 1]
# print(df0)
# print(df1)

# plot df0 & df1
plt.scatter(df0['usia'], df0['gaji'], marker='o', color='g')
plt.scatter(df1['usia'], df1['gaji'], marker='o', color='y')
plt.xlabel('Usia')
plt.ylabel('Gaji')
plt.grid(True)

# centroid
print(model.cluster_centers_)
plt.scatter(
    model.cluster_centers_[:,0],
    model.cluster_centers_[:,1],
    marker='*',
    color='r',
    s = 200
)

plt.show()