import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
df = pd.read_excel('data.xlsx')
# print(df.head(2))

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

# n_cluster = sum square error
sse_result = []
for x in range(1, 11):
    model = KMeans(n_clusters = x)
    model.fit(df[['usia', 'gaji']])
    sse_result.append(model.inertia_)

print(sse_result)

# elbow method
plt.plot(
    np.arange(1,11),
    sse_result,
    'b-'
)
plt.xlabel('Range')
plt.ylabel('Sum of squared error')
plt.grid(True)
plt.show()