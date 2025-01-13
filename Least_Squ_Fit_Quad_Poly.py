import numpy as np
import matplotlib.pyplot as plt

# Define matrix A and vector b
A = np.array([[0, 3], [1, 3], [2, 3]])
b = np.array([1, 2, 0])

# Create feature matrix with constant, linear, and quadratic terms
ones = np.ones(A.shape[0])
x_feature = A[:, 0]
squared_feature = x_feature ** 2
features = np.column_stack((ones, x_feature, squared_feature))

# Calculate the determinant of the original feature matrix
m_det = np.linalg.det(features)
if np.isclose(m_det, 0):
    raise ValueError("The determinant of the feature matrix is zero, indicating a singular matrix.")

# Calculate determinants by replacing each column with vector b, using Cramer's Rule
determinants = []
for i in range(features.shape[1]):
    modified_features = features.copy()
    modified_features[:, i] = b  # Replace column i with vector b
    det = np.linalg.det(modified_features)
    determinants.append(det / m_det)

# Coefficients of the quadratic fit
coefficients = np.array(determinants)
print("Coefficients:", coefficients)

# Plot the data points
plt.scatter(x_feature, b, label="Data Points", color="blue")

# Plot the fitted curve using the calculated coefficients
u = np.linspace(0, 2, 100)
fitted_curve = coefficients[2] * u**2 + coefficients[1] * u + coefficients[0]
plt.plot(u, fitted_curve, label="Fitted Curve (Cramer's Rule)", color="green")

# Compare with numpy's polyfit
polyfit_coeffs = np.polyfit(x_feature, b, 2)
plt.plot(u, np.polyval(polyfit_coeffs, u), 'r--', label="Polyfit Curve (Degree 2)")

# Final plot adjustments
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Quadratic Fit using Cramer's Rule and Polyfit")
plt.show()
