# Hypothesis Testing

Hypothesis testing is a statistical method for making decisions based on data.

## Basic Framework

1. **Null Hypothesis (H₀)**: The default assumption
2. **Alternative Hypothesis (H₁)**: What we want to test
3. **Test Statistic**: Measure computed from data
4. **p-value**: Probability of observing data if H₀ is true
5. **Significance Level (α)**: Threshold for rejection (typically 0.05)

**Decision Rule**: Reject H₀ if p-value < α

## Types of Errors

- **Type I Error**: Rejecting H₀ when it's true (false positive)
  - P(Type I Error) = α
- **Type II Error**: Failing to reject H₀ when it's false (false negative)
  - P(Type II Error) = β
- **Power**: 1 - β (probability of correctly rejecting false H₀)

## One-Sample Tests

### Z-Test for Mean (Known σ)
```python
import numpy as np
from scipy import stats

# Test if mean = μ₀
data = np.random.normal(5.5, 2, size=100)
mu_0 = 5.0
sigma = 2.0
alpha = 0.05

sample_mean = np.mean(data)
n = len(data)

# Test statistic
z = (sample_mean - mu_0) / (sigma / np.sqrt(n))

# p-value (two-tailed)
p_value = 2 * (1 - stats.norm.cdf(abs(z)))

print(f"Z-statistic: {z:.3f}")
print(f"p-value: {p_value:.3f}")
print(f"Reject H₀: {p_value < alpha}")
```

### T-Test for Mean (Unknown σ)
```python
# One-sample t-test
data = np.random.normal(5.5, 2, size=100)
mu_0 = 5.0

t_stat, p_value = stats.ttest_1samp(data, mu_0)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")
print(f"Reject H₀: {p_value < 0.05}")
```

### Z-Test for Proportion
```python
# Test if proportion = p₀
successes = 65
n = 100
p_0 = 0.5

p_hat = successes / n

# Test statistic
z = (p_hat - p_0) / np.sqrt(p_0 * (1 - p_0) / n)

# p-value (two-tailed)
p_value = 2 * (1 - stats.norm.cdf(abs(z)))

print(f"Sample proportion: {p_hat}")
print(f"Z-statistic: {z:.3f}")
print(f"p-value: {p_value:.3f}")
```

## Two-Sample Tests

### Independent Samples T-Test
```python
# Compare means of two independent groups
group1 = np.random.normal(5, 2, size=50)
group2 = np.random.normal(5.5, 2, size=50)

# Equal variances assumed
t_stat, p_value = stats.ttest_ind(group1, group2)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")

# Unequal variances (Welch's t-test)
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
print(f"Welch's t-statistic: {t_stat:.3f}")
```

### Paired Samples T-Test
```python
# Compare means of paired observations
before = np.random.normal(100, 15, size=30)
after = before + np.random.normal(5, 10, size=30)

t_stat, p_value = stats.ttest_rel(before, after)

print(f"Mean difference: {np.mean(after - before):.3f}")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")
```

### Two-Sample Proportion Test
```python
# Compare two proportions
successes1, n1 = 45, 100
successes2, n2 = 60, 100

p1 = successes1 / n1
p2 = successes2 / n2

# Pooled proportion
p_pool = (successes1 + successes2) / (n1 + n2)

# Test statistic
z = (p1 - p2) / np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

# p-value
p_value = 2 * (1 - stats.norm.cdf(abs(z)))

print(f"Difference in proportions: {p1 - p2:.3f}")
print(f"Z-statistic: {z:.3f}")
print(f"p-value: {p_value:.3f}")
```

## Analysis of Variance (ANOVA)

### One-Way ANOVA
```python
# Compare means of multiple groups
group1 = np.random.normal(5, 2, size=30)
group2 = np.random.normal(5.5, 2, size=30)
group3 = np.random.normal(6, 2, size=30)

f_stat, p_value = stats.f_oneway(group1, group2, group3)

print(f"F-statistic: {f_stat:.3f}")
print(f"p-value: {p_value:.3f}")
```

### Two-Way ANOVA
```python
import pandas as pd
from scipy.stats import f_oneway

# Create sample data
np.random.seed(42)
data = pd.DataFrame({
    'value': np.random.randn(120),
    'factor_a': np.repeat(['A1', 'A2'], 60),
    'factor_b': np.tile(np.repeat(['B1', 'B2', 'B3'], 20), 2)
})

# Using statsmodels for two-way ANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

model = ols('value ~ C(factor_a) + C(factor_b) + C(factor_a):C(factor_b)', data=data).fit()
anova_table = anova_lm(model, typ=2)
print(anova_table)
```

## Non-Parametric Tests

### Mann-Whitney U Test
```python
# Non-parametric alternative to independent t-test
group1 = np.random.exponential(2, size=50)
group2 = np.random.exponential(2.5, size=50)

u_stat, p_value = stats.mannwhitneyu(group1, group2)

print(f"U-statistic: {u_stat:.3f}")
print(f"p-value: {p_value:.3f}")
```

### Wilcoxon Signed-Rank Test
```python
# Non-parametric alternative to paired t-test
before = np.random.exponential(2, size=30)
after = before * np.random.uniform(0.8, 1.2, size=30)

w_stat, p_value = stats.wilcoxon(before, after)

print(f"W-statistic: {w_stat:.3f}")
print(f"p-value: {p_value:.3f}")
```

### Kruskal-Wallis H Test
```python
# Non-parametric alternative to one-way ANOVA
group1 = np.random.exponential(2, size=30)
group2 = np.random.exponential(2.5, size=30)
group3 = np.random.exponential(3, size=30)

h_stat, p_value = stats.kruskal(group1, group2, group3)

print(f"H-statistic: {h_stat:.3f}")
print(f"p-value: {p_value:.3f}")
```

## Chi-Square Tests

### Chi-Square Goodness of Fit
```python
# Test if observed frequencies match expected
observed = np.array([18, 22, 20, 15, 25])
expected = np.array([20, 20, 20, 20, 20])

chi2_stat, p_value = stats.chisquare(observed, expected)

print(f"χ² statistic: {chi2_stat:.3f}")
print(f"p-value: {p_value:.3f}")
```

### Chi-Square Test of Independence
```python
# Test independence in contingency table
observed = np.array([[10, 20, 30],
                     [15, 25, 35],
                     [20, 30, 40]])

chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)

print(f"χ² statistic: {chi2_stat:.3f}")
print(f"p-value: {p_value:.3f}")
print(f"Degrees of freedom: {dof}")
```

## Variance Tests

### F-Test for Equal Variances
```python
group1 = np.random.normal(5, 2, size=50)
group2 = np.random.normal(5, 3, size=50)

var1 = np.var(group1, ddof=1)
var2 = np.var(group2, ddof=1)

f_stat = var1 / var2
df1, df2 = len(group1) - 1, len(group2) - 1
p_value = 2 * min(stats.f.cdf(f_stat, df1, df2),
                  1 - stats.f.cdf(f_stat, df1, df2))

print(f"F-statistic: {f_stat:.3f}")
print(f"p-value: {p_value:.3f}")
```

### Levene's Test
```python
# Robust test for equal variances
stat, p_value = stats.levene(group1, group2)

print(f"Levene statistic: {stat:.3f}")
print(f"p-value: {p_value:.3f}")
```

## Multiple Testing Correction

### Bonferroni Correction
```python
from statsmodels.stats.multitest import multipletests

# Multiple tests
p_values = np.array([0.01, 0.04, 0.03, 0.08, 0.02])

# Bonferroni correction
reject, p_corrected, _, _ = multipletests(p_values, method='bonferroni')

print(f"Original p-values: {p_values}")
print(f"Corrected p-values: {p_corrected}")
print(f"Reject H₀: {reject}")
```

### False Discovery Rate (FDR)
```python
# Benjamini-Hochberg procedure
reject, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')

print(f"FDR-corrected p-values: {p_corrected}")
print(f"Reject H₀: {reject}")
```

## Power Analysis

```python
from statsmodels.stats.power import ttest_power, tt_solve_power

# Calculate power for t-test
effect_size = 0.5  # Cohen's d
alpha = 0.05
n = 50

power = ttest_power(effect_size, n, alpha)
print(f"Power: {power:.3f}")

# Calculate required sample size for desired power
required_n = tt_solve_power(effect_size, power=0.8, alpha=alpha)
print(f"Required sample size: {int(np.ceil(required_n))}")
```
