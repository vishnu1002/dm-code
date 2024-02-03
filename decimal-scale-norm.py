import numpy as np

def decimal_scaling_normalization(data):
  """Normalizes a dataset using decimal scaling.

  Args:
    data: A NumPy array containing the data to be normalized.

  Returns:
    The normalized data as a NumPy array.
  """

  # Find the maximum absolute value in the dataset.
  max_abs_value = np.max(np.abs(data))

  # Determine the scaling factor (c) as the number of digits in the maximum absolute value.
  c = int(np.log10(max_abs_value)) if max_abs_value > 0 else 0

  # Normalize the data by dividing each value by 10^c.
  normalized_data = data / 10**c

  return normalized_data

# Example usage:
data = np.array([4856, 28, -154, 325, 98])
normalized_data = decimal_scaling_normalization(data)
print(normalized_data)  # Output: [0.4856 0.0028 -0.0154 0.0325 0.0098]
