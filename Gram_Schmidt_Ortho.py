# Input: Number of vectors
k = int(input("Enter the number of vectors (k): "))

# Step 1: Gather the input vectors
input_vectors = []
for i in range(k):
    vector = [float(x) for x in input(f"Enter vector {i + 1} as space-separated values: ").split()]
    input_vectors.append(vector)
    print(f"Vector {i + 1}: {vector}")

# Step 2: Perform Gram-Schmidt Orthogonalization
orthogonal_basis = []
orthonormal_sets = []

for vector in input_vectors:
    for basis_vector in orthogonal_basis:
        # Calculate projection of vector onto basis_vector
        dot_product = sum(v1 * v2 for v1, v2 in zip(vector, basis_vector))
        projection = [dot_product / sum(vi ** 2 for vi in basis_vector) * bi for bi in basis_vector]
        # Subtract projection from vector
        vector = [v - p for v, p in zip(vector, projection)]
    orthogonal_basis.append(vector)

# Step 3: Normalize the orthogonal basis to get orthonormal sets
for basis_vector in orthogonal_basis:
    magnitude = sum(x ** 2 for x in basis_vector) ** 0.5
    if magnitude != 0:  # Prevent division by zero for zero vectors
        orthonormal_sets.append([x / magnitude for x in basis_vector])

# Output Results
print(f"\nThe orthogonal basis vectors are: {orthogonal_basis}")
print(f"The orthonormal sets are: {orthonormal_sets}")
