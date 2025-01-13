import numpy as np


def eigen_decomposition(A, max_iterations, tolerance):
    n = A.shape[0]
    eigenvectors = np.eye(n)
    eigenvalues = np.zeros(n)

    for k in range(n):
        # Start with a random vector
        x = np.random.randn(n)
        x = x / np.linalg.norm(x)

        # Power iteration
        for iteration in range(max_iterations):
            y = A @ x
            lambda_ = x @ y
            x = y / np.linalg.norm(y)

            # Check convergence
            if np.linalg.norm(A @ x - lambda_ * x) < tolerance:
                break

        eigenvalues[k] = lambda_
        eigenvectors[:, k] = x

        # Deflate the matrix
        A = A - lambda_ * np.outer(x, x)

    return eigenvalues, eigenvectors


# Simplified matrix input
def input_matrix():
    print("Enter the matrix row by row, elements separated by spaces.")
    rows = int(input("How many rows in the matrix? "))
    matrix = []

    for i in range(rows):
        row = list(map(float, input(f"Enter row {i + 1}: ").split()))
        matrix.append(row)

    return np.array(matrix)


# Get input matrix
A = input_matrix()

max_iterations = 1000
tolerance = 1e-6

# Perform eigen decomposition
eigenvalues, eigenvectors = eigen_decomposition(A, max_iterations, tolerance)

# Output results
print('\nEigenvectors:')
print(eigenvectors)
print('Eigenvalues:')
print(eigenvalues)
