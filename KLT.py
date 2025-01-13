import numpy as np
import matplotlib.pyplot as plt

# Define a sample dataset (2D points)
X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
])

# Step 1: Center the data
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# Step 2: Compute the covariance matrix
cov_matrix = np.cov(X_centered.T)

# Step 3: Perform eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Project the data onto eigenvectors
X_transformed = X_centered @ eigenvectors

# Display results
print("Original Data:\n", X)
print("\nMean of Data:\n", X_mean)
print("\nCovariance Matrix:\n", cov_matrix)
print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)
print("\nTransformed Data:\n", X_transformed)

# Plot original and transformed data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], color='blue', label='Original Data')
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], color='red', label='KLT Transformed Data')
plt.title("Karhunen-Loève Transform (KLT)")
plt.xlabel("First Feature")
plt.ylabel("Second Feature")
plt.legend()
plt.grid(True)
plt.show()
