import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
df = pd.read_excel('data.xlsx')
# print(df.head(2))

# plot data
plt.scatter(df['usia'], df['gaji'])
plt.xlabel('Gaji')
plt.ylabel('Usia')
plt.grid(True)
plt.show()