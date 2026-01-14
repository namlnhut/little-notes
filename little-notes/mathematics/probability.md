# Probability

Probability theory provides the mathematical framework for reasoning about uncertainty.

## Basic Concepts

### Probability Axioms
1. $0 \leq P(A) \leq 1$ for any event $A$
2. $P(\Omega) = 1$ (probability of sample space)
3. For disjoint events: $P(A \cup B) = P(A) + P(B)$

### Conditional Probability
$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

### Bayes' Theorem
$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

## Random Variables

### Discrete Random Variables
```python
import numpy as np
import matplotlib.pyplot as plt

# Probability mass function (PMF)
# Example: Fair die
outcomes = np.arange(1, 7)
pmf = np.ones(6) / 6

plt.bar(outcomes, pmf)
plt.xlabel('Outcome')
plt.ylabel('Probability')
plt.title('Fair Die PMF')
plt.show()

# Expected value
E_X = np.sum(outcomes * pmf)
print(f"Expected value: {E_X}")
```

### Continuous Random Variables
```python
# Probability density function (PDF)
from scipy.stats import norm

# Standard normal distribution
x = np.linspace(-4, 4, 1000)
pdf = norm.pdf(x, loc=0, scale=1)

plt.plot(x, pdf)
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Standard Normal PDF')
plt.show()
```

## Common Distributions

### Bernoulli Distribution
```python
from scipy.stats import bernoulli

# Coin flip: p = 0.5
p = 0.5
samples = bernoulli.rvs(p, size=1000)
print(f"Mean: {samples.mean()}")
print(f"Variance: {samples.var()}")
```

### Binomial Distribution
```python
from scipy.stats import binom

# Number of heads in n coin flips
n, p = 10, 0.5
x = np.arange(0, n+1)
pmf = binom.pmf(x, n, p)

plt.bar(x, pmf)
plt.xlabel('Number of successes')
plt.ylabel('Probability')
plt.title(f'Binomial({n}, {p})')
plt.show()
```

### Poisson Distribution
```python
from scipy.stats import poisson

# Events occurring at rate λ
lambda_ = 3
x = np.arange(0, 15)
pmf = poisson.pmf(x, lambda_)

plt.bar(x, pmf)
plt.xlabel('Number of events')
plt.ylabel('Probability')
plt.title(f'Poisson({lambda_})')
plt.show()
```

### Normal (Gaussian) Distribution
```python
from scipy.stats import norm

# N(μ, σ²)
mu, sigma = 0, 1
x = np.linspace(-4, 4, 1000)
pdf = norm.pdf(x, mu, sigma)

plt.plot(x, pdf)
plt.xlabel('x')
plt.ylabel('Density')
plt.title(f'Normal({mu}, {sigma}²)')
plt.show()

# Generate samples
samples = norm.rvs(mu, sigma, size=1000)
```

### Exponential Distribution
```python
from scipy.stats import expon

# Time between events
lambda_ = 1.5
x = np.linspace(0, 5, 1000)
pdf = expon.pdf(x, scale=1/lambda_)

plt.plot(x, pdf)
plt.xlabel('Time')
plt.ylabel('Density')
plt.title('Exponential Distribution')
plt.show()
```

## Joint Distributions

### Covariance and Correlation
```python
# Generate correlated data
mean = [0, 0]
cov = [[1, 0.8],
       [0.8, 1]]
X = np.random.multivariate_normal(mean, cov, size=1000)

# Compute covariance
cov_matrix = np.cov(X.T)
print(f"Covariance matrix:\n{cov_matrix}")

# Compute correlation
corr_matrix = np.corrcoef(X.T)
print(f"Correlation matrix:\n{corr_matrix}")

# Visualize
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Correlated Variables')
plt.show()
```

### Multivariate Normal Distribution
```python
from scipy.stats import multivariate_normal

mean = np.array([1, 2])
cov = np.array([[1, 0.5],
                [0.5, 2]])

# PDF evaluation
rv = multivariate_normal(mean, cov)
x = np.array([1.5, 2.5])
pdf_value = rv.pdf(x)

# Generate samples
samples = rv.rvs(size=1000)
```

## Law of Large Numbers

```python
# Simulate coin flips
n_flips = 10000
flips = np.random.binomial(1, 0.5, size=n_flips)
cumulative_mean = np.cumsum(flips) / np.arange(1, n_flips + 1)

plt.plot(cumulative_mean)
plt.axhline(y=0.5, color='r', linestyle='--', label='True mean')
plt.xlabel('Number of flips')
plt.ylabel('Cumulative mean')
plt.title('Law of Large Numbers')
plt.legend()
plt.show()
```

## Central Limit Theorem

```python
# Sum of uniform random variables approaches normal
n_samples = 10000
n_sum = 30

# Generate samples
samples = np.random.uniform(0, 1, size=(n_samples, n_sum))
sample_means = samples.mean(axis=1)

# Plot histogram
plt.hist(sample_means, bins=50, density=True, alpha=0.7)

# Overlay theoretical normal
mu = 0.5
sigma = np.sqrt(1/12 / n_sum)
x = np.linspace(0.3, 0.7, 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2)
plt.title('Central Limit Theorem')
plt.show()
```

## Monte Carlo Simulation

### Estimating π
```python
# Estimate π using random sampling
n_samples = 100000
x = np.random.uniform(-1, 1, n_samples)
y = np.random.uniform(-1, 1, n_samples)

# Check if inside unit circle
inside = (x**2 + y**2) <= 1
pi_estimate = 4 * inside.sum() / n_samples

print(f"Estimated π: {pi_estimate}")
print(f"Actual π: {np.pi}")
print(f"Error: {abs(pi_estimate - np.pi)}")
```

### Integration via Monte Carlo
```python
def monte_carlo_integrate(f, a, b, n_samples=10000):
    """Estimate integral using Monte Carlo"""
    x = np.random.uniform(a, b, n_samples)
    return (b - a) * np.mean(f(x))

# Example: integrate x^2 from 0 to 1
f = lambda x: x**2
estimate = monte_carlo_integrate(f, 0, 1)
exact = 1/3
print(f"Estimate: {estimate}")
print(f"Exact: {exact}")
```

## Applications in Machine Learning

### Maximum Likelihood Estimation
```python
# Estimate normal distribution parameters
data = np.random.normal(5, 2, size=1000)

# MLE for normal distribution
mu_mle = np.mean(data)
sigma_mle = np.std(data, ddof=0)

print(f"Estimated μ: {mu_mle}")
print(f"Estimated σ: {sigma_mle}")
```

### Bayesian Inference
```python
# Beta-Binomial conjugate prior
from scipy.stats import beta

# Prior: Beta(α, β)
alpha_prior, beta_prior = 2, 2

# Observed data: k successes in n trials
n, k = 10, 7

# Posterior: Beta(α + k, β + n - k)
alpha_post = alpha_prior + k
beta_post = beta_prior + n - k

# Plot prior and posterior
x = np.linspace(0, 1, 100)
plt.plot(x, beta.pdf(x, alpha_prior, beta_prior), label='Prior')
plt.plot(x, beta.pdf(x, alpha_post, beta_post), label='Posterior')
plt.xlabel('θ')
plt.ylabel('Density')
plt.legend()
plt.title('Bayesian Inference')
plt.show()
```
