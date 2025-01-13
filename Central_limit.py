import random
import numpy as np
import matplotlib.pyplot as plt
import math

# Parameters for the uniform distribution
lower_bound, upper_bound = 0, 1  # Uniform distribution bounds
num_sums = 10000  # Number of sums to generate
n_values = [5, 10, 30, 50]  # Different numbers of uniform variables to sum


# Function to compute the PDF of a normal distribution
def normal_pdf(x, mean, std_dev):
    return (1 / (std_dev * math.sqrt(2 * math.pi))) * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))


# Plot the distributions of the sums
plt.figure(figsize=(12, 8))

for n in n_values:
    # Generate sums of n uniform random variables
    sums = [sum(random.uniform(lower_bound, upper_bound) for _ in range(n)) for _ in range(num_sums)]

    # Calculate mean and standard deviation of the sums
    mean_sum = np.mean(sums)
    std_dev_sum = np.std(sums)

    # Plot the histogram of sums
    plt.hist(sums, bins=30, density=True, alpha=0.6, label=f'Sum of {n} Uniform RVs')

    # Plot the theoretical normal PDF
    x = np.linspace(min(sums), max(sums), 1000)
    pdf = normal_pdf(x, mean_sum, std_dev_sum)
    plt.plot(x, pdf, linewidth=2, label=f'Normal PDF (n={n})')

# Add titles and labels
plt.title('Verification of Central Limit Theorem with Uniform Distribution')
plt.xlabel('Sum of Random Variables')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
