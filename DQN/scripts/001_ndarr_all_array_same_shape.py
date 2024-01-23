




import numpy as np

# Assuming lys_ls is a list containing 2D NumPy arrays
# Example of lys_ls:
lys_ls = [[np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])],[np.array([[1, 2,3], [3, 4,3]]), np.array([[5, 6,3], [7, 8,3]])]]

# Convert the list of arrays into a 3D array
combined_array = np.stack(lys_ls)
print(combined_array)

















