import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Define input array
A = np.array([[3.1, 2.3], [2.3, 4.2], [3.9, 3.5], [3.7, 6.4], [4.8, 1.9], [8.3, 3.1], [5.2, 7.5], [4.8, 4.7], [3.5, 5.1], [4.4, 2.9]])

# Define number of neighbors 
k = 3

# Define query point
test_data = [3.3, 2.9]

plt.figure()
plt.title('Input data')
# A[:, 0] means all rows, A[:, 1] means all columns
# marker='o' means circle, s=100 means size of circle, color='black' means color of circle
plt.scatter(A[:, 0], A[:, 1], marker='o', s=100, color='black')

# Auto means automatically choose the best algorithm
# .fit(A) means fit the model using A as training data
knn_model = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(A)

# .kneighbors([test_data]) means Finds the K-neighbors of a point
# distances means distances to the nearest neighbors of each point
# indices means indices of the nearest points in the population matrix, which is 0,9,1
distances, indices = knn_model.kneighbors([test_data])

print("\nK Nearest Neighbors:")

# Rank the neighbors by distance and print them, index is the index of the nearest points
# enumerate means to iterate over indices[0][:k] and start from 1
# :k means slicing the subarray from 0 to k, k is 3 so it is the first 3 nearest points
for rank, index in enumerate(indices[0][:k], start=1):
    print(str(rank) + " is", A[index])

plt.figure()
plt.title('Nearest neighbors')
plt.scatter(A[:, 0], A[:, 1], marker='o', s=100, color='k')
# [0] means the first element of indices
# [:] means all rows, but 2d array so can be omitted
plt.scatter(A[indices][0][:][:, 0], A[indices][0][:][:, 1], marker='o', s=250, color='k', facecolors='none')
plt.scatter(test_data[0], test_data[1], marker='x', s=100, color='k')
plt.show()
