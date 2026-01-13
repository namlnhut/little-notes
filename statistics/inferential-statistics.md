# Inferential Statistics

Inferential statistics uses sample data to make inferences about populations.

## Sampling Distributions

### Central Limit Theorem
The sampling distribution of the mean approaches normal distribution:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Population: uniform distribution
population = np.random.uniform(0, 10, size=100000)

# Draw many samples and compute means
sample_size = 30
n_samples = 1000
sample_means = []

for _ in range(n_samples):
    sample = np.random.choice(population, size=sample_size)
    sample_means.append(np.mean(sample))

sample_means = np.array(sample_means)

# Plot sampling distribution
plt.hist(sample_means, bins=50, density=True, alpha=0.7)
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.title('Sampling Distribution of the Mean')
plt.show()

print(f"Population mean: {np.mean(population):.3f}")
print(f"Mean of sample means: {np.mean(sample_means):.3f}")
```

## Confidence Intervals

### Confidence Interval for Mean (Known σ)
```python
# Z-interval when population std is known
data = np.random.normal(5, 2, size=100)
sample_mean = np.mean(data)
sigma = 2  # Known population std
n = len(data)
confidence = 0.95

# Critical value
z_critical = stats.norm.ppf((1 + confidence) / 2)

# Margin of error
margin = z_critical * (sigma / np.sqrt(n))

# Confidence interval
ci_lower = sample_mean - margin
ci_upper = sample_mean + margin

print(f"95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")
```

### Confidence Interval for Mean (Unknown σ)
```python
# T-interval when population std is unknown
sample_std = np.std(data, ddof=1)
t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)

margin = t_critical * (sample_std / np.sqrt(n))
ci_lower = sample_mean - margin
ci_upper = sample_mean + margin

print(f"95% CI (t-interval): ({ci_lower:.3f}, {ci_upper:.3f})")
```

### Confidence Interval for Proportion
```python
# Proportion confidence interval
successes = 65
n = 100
p_hat = successes / n

# Normal approximation
z_critical = stats.norm.ppf(0.975)  # 95% CI
margin = z_critical * np.sqrt(p_hat * (1 - p_hat) / n)

ci_lower = p_hat - margin
ci_upper = p_hat + margin

print(f"95% CI for proportion: ({ci_lower:.3f}, {ci_upper:.3f})")
```

## Bootstrap Resampling

Non-parametric method for estimating confidence intervals:

```python
def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, confidence=0.95):
    """Compute bootstrap confidence interval"""
    n = len(data)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Percentile method
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return ci_lower, ci_upper

# Example
data = np.random.exponential(2, size=100)
ci = bootstrap_ci(data)
print(f"Bootstrap 95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")

# Bootstrap for other statistics (e.g., median)
ci_median = bootstrap_ci(data, statistic=np.median)
print(f"Bootstrap 95% CI for median: ({ci_median[0]:.3f}, {ci_median[1]:.3f})")
```

## Point Estimation

### Method of Moments
```python
# Estimate parameters by matching moments
data = np.random.gamma(2, 2, size=1000)

# For gamma distribution: E[X] = αβ, Var[X] = αβ²
sample_mean = np.mean(data)
sample_var = np.var(data, ddof=1)

# Method of moments estimators
beta_hat = sample_var / sample_mean
alpha_hat = sample_mean / beta_hat

print(f"Estimated α: {alpha_hat:.3f}")
print(f"Estimated β: {beta_hat:.3f}")
```

### Maximum Likelihood Estimation
```python
from scipy.optimize import minimize

def normal_neg_log_likelihood(params, data):
    """Negative log-likelihood for normal distribution"""
    mu, sigma = params
    n = len(data)
    return n/2 * np.log(2*np.pi*sigma**2) + np.sum((data - mu)**2) / (2*sigma**2)

# Generate data
true_mu, true_sigma = 5, 2
data = np.random.normal(true_mu, true_sigma, size=1000)

# MLE
result = minimize(normal_neg_log_likelihood, x0=[0, 1], args=(data,))
mu_mle, sigma_mle = result.x

print(f"True parameters: μ={true_mu}, σ={true_sigma}")
print(f"MLE estimates: μ={mu_mle:.3f}, σ={sigma_mle:.3f}")
```

## Sample Size Determination

### For Estimating Mean
```python
def sample_size_mean(sigma, margin_error, confidence=0.95):
    """Calculate required sample size for estimating mean"""
    z = stats.norm.ppf((1 + confidence) / 2)
    n = (z * sigma / margin_error) ** 2
    return int(np.ceil(n))

# Example: Estimate mean with margin of error 0.5
sigma = 2
margin = 0.5
n_required = sample_size_mean(sigma, margin)
print(f"Required sample size: {n_required}")
```

### For Estimating Proportion
```python
def sample_size_proportion(p, margin_error, confidence=0.95):
    """Calculate required sample size for estimating proportion"""
    z = stats.norm.ppf((1 + confidence) / 2)
    n = p * (1 - p) * (z / margin_error) ** 2
    return int(np.ceil(n))

# Conservative estimate (p = 0.5 gives maximum)
n_required = sample_size_proportion(0.5, 0.05)
print(f"Required sample size: {n_required}")
```

## Prediction Intervals

```python
# Prediction interval for new observation
from sklearn.linear_model import LinearRegression

X = np.random.randn(100, 1)
y = 2*X.ravel() + np.random.randn(100)*0.5

model = LinearRegression()
model.fit(X, y)

# Predict new point
X_new = np.array([[1.5]])
y_pred = model.predict(X_new)[0]

# Prediction interval
residuals = y - model.predict(X)
residual_std = np.std(residuals, ddof=2)
n = len(X)

# Assuming normal errors
t_val = stats.t.ppf(0.975, df=n-2)
margin = t_val * residual_std * np.sqrt(1 + 1/n)

pi_lower = y_pred - margin
pi_upper = y_pred + margin

print(f"Prediction: {y_pred:.3f}")
print(f"95% Prediction Interval: ({pi_lower:.3f}, {pi_upper:.3f})")
```

## Bayesian Inference

```python
# Bayesian updating: Beta-Binomial conjugate
from scipy.stats import beta

# Prior: Beta(α, β)
alpha_prior, beta_prior = 1, 1  # Uniform prior

# Observed data
n_trials = 20
n_successes = 15

# Posterior: Beta(α + successes, β + failures)
alpha_post = alpha_prior + n_successes
beta_post = beta_prior + (n_trials - n_successes)

# Posterior mean (point estimate)
posterior_mean = alpha_post / (alpha_post + beta_post)
print(f"Posterior mean: {posterior_mean:.3f}")

# Credible interval (Bayesian CI)
credible_interval = beta.interval(0.95, alpha_post, beta_post)
print(f"95% Credible Interval: ({credible_interval[0]:.3f}, {credible_interval[1]:.3f})")

# Visualize
x = np.linspace(0, 1, 1000)
plt.plot(x, beta.pdf(x, alpha_prior, beta_prior), label='Prior')
plt.plot(x, beta.pdf(x, alpha_post, beta_post), label='Posterior')
plt.xlabel('θ')
plt.ylabel('Density')
plt.legend()
plt.title('Bayesian Inference')
plt.show()
```
