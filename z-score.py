import numpy as np
data = np.array([6, 7, 7, 12, 13, 13, 15, 16, 19, 22])
# Calculate mean and standard deviation
mean = np.mean(data)
std = np.std(data)
# Normalize the data
normalized_data = (data - mean) / std
print(normalized_data)


#using pandas:
import pandas as pd
df = pd.DataFrame({'col1': [6, 7, 7, 12, 13, 13, 15, 16, 19, 22]})
# Normalize the column
df['col1_zscore'] = (df['col1'] - df['col1'].mean()) / df['col1'].std()
print(df)

#using scikit
from sklearn.preprocessing import StandardScaler
data = np.array([6, 7, 7, 12, 13, 13, 15, 16, 19, 22])
# Create a scaler object
scaler = StandardScaler()
# Fit the scaler to the data
scaler.fit(data.reshape(-1, 1))  # Reshape for a single feature
# Transform the data
normalized_data = scaler.transform(data.reshape(-1, 1))
print(normalized_data)
