import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom, geom, poisson, uniform, norm, expon, laplace

# Parameters for the distributions
n = 1000  # Number of samples
p_bernoulli = 0.5
n_trials_binomial, p_binomial = 10, 0.5
p_geometric = 0.5
lambda_poisson = 4
a_uniform, b_uniform = 0, 1
mu_gaussian, sigma_gaussian = 0, 1
lambda_exponential = 1
mu_laplacian, b_laplacian = 0, 1

# Generate random samples
samples = {
    "Bernoulli": np.random.choice([0, 1], size=n, p=[1-p_bernoulli, p_bernoulli]),
    "Binomial": np.random.binomial(n_trials_binomial, p_binomial, size=n),
    "Geometric": np.random.geometric(p_geometric, size=n),
    "Poisson": np.random.poisson(lambda_poisson, size=n),
    "Uniform": np.random.uniform(a_uniform, b_uniform, size=n),
    "Gaussian": np.random.normal(mu_gaussian, sigma_gaussian, size=n),
    "Exponential": np.random.exponential(1/lambda_exponential, size=n),
    "Laplacian": np.random.laplace(mu_laplacian, b_laplacian, size=n),
}

# x-values for PDFs/PMFs
x_values = {
    "Bernoulli": np.array([0, 1]),
    "Binomial": np.arange(0, n_trials_binomial + 1),
    "Geometric": np.arange(1, 15),
    "Poisson": np.arange(0, 20),
    "Uniform": np.linspace(a_uniform, b_uniform, 100),
    "Gaussian": np.linspace(mu_gaussian - 4 * sigma_gaussian, mu_gaussian + 4 * sigma_gaussian, 100),
    "Exponential": np.linspace(0, 5 / lambda_exponential, 100),
    "Laplacian": np.linspace(mu_laplacian - 4 * b_laplacian, mu_laplacian + 4 * b_laplacian, 100),
}

# Compute theoretical PDFs/PMFs
theoretical = {
    "Bernoulli": bernoulli.pmf(x_values["Bernoulli"], p_bernoulli),
    "Binomial": binom.pmf(x_values["Binomial"], n_trials_binomial, p_binomial),
    "Geometric": geom.pmf(x_values["Geometric"], p_geometric),
    "Poisson": poisson.pmf(x_values["Poisson"], lambda_poisson),
    "Uniform": uniform.pdf(x_values["Uniform"], a_uniform, b_uniform - a_uniform),
    "Gaussian": norm.pdf(x_values["Gaussian"], mu_gaussian, sigma_gaussian),
    "Exponential": expon.pdf(x_values["Exponential"], scale=1 / lambda_exponential),
    "Laplacian": laplace.pdf(x_values["Laplacian"], mu_laplacian, b_laplacian),
}

# Plot histograms with overlays
plt.figure(figsize=(15, 12))
for i, (name, data) in enumerate(samples.items(), 1):
    plt.subplot(4, 2, i)
    plt.hist(data, bins=30, density=True, alpha=0.6, color='lightblue', label="Histogram")
    x = x_values[name]
    pdf_pmf = theoretical[name]
    if name in ["Bernoulli", "Binomial", "Geometric", "Poisson"]:  # Discrete distributions
        plt.stem(x, pdf_pmf, basefmt=" ", linefmt="r-", markerfmt="ro", label="PMF")
    else:  # Continuous distributions
        plt.plot(x, pdf_pmf, 'r-', label="PDF")
    plt.title(name)
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Density/Probability")
plt.tight_layout()
plt.show()
