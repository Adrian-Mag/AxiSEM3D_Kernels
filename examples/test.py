import numpy as np

# Create a 2D numpy array
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Create index arrays that contain all indices
row_indices = np.array([0, 1, 2])
col_indices = np.array([0, 1, 2])

# Use the index arrays to select all rows and columns from the original array
selected_rows = arr[row_indices]
selected_cols = arr[:, col_indices]
# This will output the diagonal elements of arr
print(arr[row_indices][:, col_indices])