#manual:
import numpy as np

def min_max_scale(X):
    """Scales features of X to a given range [0, 1] using min-max normalization."""
    min_max_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    X_scaled = min_max_scaler(X)
    return X_scaled

# Example usage:
data = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
normalized_data = min_max_scale(data)
print(normalized_data)


#scikit
from sklearn import preprocessing

# Create a MinMaxScaler object
scaler = preprocessing.MinMaxScaler()

# Fit the scaler to the data
scaler.fit(data)

# Transform the data using the fitted scaler
normalized_data = scaler.transform(data)
print(normalized_data)
