import numpy as np


# Function to get matrix input from the user
def get_matrix():
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))

    print("Enter the matrix elements row by row (separated by spaces):")
    matrix = []
    for i in range(rows):
        row = list(map(float, input(f"Row {i + 1}: ").split()))
        if len(row) != cols:
            print("Error: Number of elements does not match the number of columns.")
            return None
        matrix.append(row)

    return np.array(matrix)


# Get matrix input
A = get_matrix()
if A is None:
    print("Invalid matrix input.")
else:
    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(A)

    # Display the results
    print("\nMatrix A:")
    print(A)
    print("\nLeft singular vectors (U):")
    print(U)
    print("\nSingular values (Sigma):")
    print(S)
    print("\nRight singular vectors (V^T):")
    print(Vt)

    # Reconstruct the original matrix using U, Sigma, and Vt
    Sigma = np.zeros((A.shape[0], A.shape[1]))
    np.fill_diagonal(Sigma, S)
    A_reconstructed = U @ Sigma @ Vt

    print("\nReconstructed Matrix A (using U, Sigma, V^T):")
    print(A_reconstructed)
