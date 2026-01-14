# Descriptive Statistics

Descriptive statistics summarize and describe the main features of a dataset.

## Measures of Central Tendency

### Mean
The average value:
```python
import numpy as np

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mean = np.mean(data)
print(f"Mean: {mean}")
```

### Median
The middle value when sorted:
```python
median = np.median(data)
print(f"Median: {median}")

# Median is robust to outliers
data_with_outlier = np.append(data, 1000)
print(f"Mean with outlier: {np.mean(data_with_outlier)}")
print(f"Median with outlier: {np.median(data_with_outlier)}")
```

### Mode
The most frequently occurring value:
```python
from scipy import stats

data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])
mode = stats.mode(data, keepdims=True)
print(f"Mode: {mode.mode[0]}")
```

## Measures of Dispersion

### Variance and Standard Deviation
```python
# Variance: average squared deviation from mean
variance = np.var(data)
print(f"Population variance: {variance}")

# Sample variance (Bessel's correction)
sample_variance = np.var(data, ddof=1)
print(f"Sample variance: {sample_variance}")

# Standard deviation: square root of variance
std = np.std(data)
print(f"Standard deviation: {std}")
```

### Range
```python
data_range = np.ptp(data)  # peak-to-peak
print(f"Range: {data_range}")
```

### Interquartile Range (IQR)
```python
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
print(f"IQR: {iqr}")

# Detect outliers
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = data[(data < lower_bound) | (data > upper_bound)]
print(f"Outliers: {outliers}")
```

## Measures of Shape

### Skewness
Measure of asymmetry:
```python
from scipy.stats import skew

# Positive skew: right tail
# Negative skew: left tail
# Zero skew: symmetric
skewness = skew(data)
print(f"Skewness: {skewness}")
```

### Kurtosis
Measure of tail heaviness:
```python
from scipy.stats import kurtosis

# Positive: heavy tails (leptokurtic)
# Negative: light tails (platykurtic)
# Zero: normal-like tails (mesokurtic)
kurt = kurtosis(data)
print(f"Kurtosis: {kurt}")
```

## Quantiles and Percentiles

```python
# Quartiles
q0 = np.percentile(data, 0)   # Minimum
q1 = np.percentile(data, 25)  # First quartile
q2 = np.percentile(data, 50)  # Median
q3 = np.percentile(data, 75)  # Third quartile
q4 = np.percentile(data, 100) # Maximum

print(f"Five-number summary: {q0}, {q1}, {q2}, {q3}, {q4}")

# Arbitrary percentiles
p90 = np.percentile(data, 90)
print(f"90th percentile: {p90}")
```

## Visualization

### Histogram
```python
import matplotlib.pyplot as plt

data = np.random.normal(0, 1, 1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()
```

### Box Plot
```python
# Box plot shows five-number summary
plt.figure(figsize=(8, 6))
plt.boxplot(data, vert=True)
plt.ylabel('Value')
plt.title('Box Plot')
plt.show()
```

### Q-Q Plot
Check if data follows a distribution:
```python
from scipy import stats

fig, ax = plt.subplots(figsize=(8, 6))
stats.probplot(data, dist="norm", plot=ax)
plt.title('Q-Q Plot')
plt.show()
```

## Correlation and Covariance

### Covariance
```python
x = np.random.randn(100)
y = 2*x + np.random.randn(100)*0.5

# Covariance
cov_matrix = np.cov(x, y)
print(f"Covariance matrix:\n{cov_matrix}")
```

### Correlation Coefficient
```python
# Pearson correlation: -1 to 1
correlation = np.corrcoef(x, y)[0, 1]
print(f"Correlation: {correlation}")

# Spearman rank correlation (non-parametric)
spearman_corr, p_value = stats.spearmanr(x, y)
print(f"Spearman correlation: {spearman_corr}")
```

### Scatter Plot with Correlation
```python
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Scatter Plot (r = {correlation:.3f})')
plt.show()
```

## Summary Statistics with Pandas

```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'A': np.random.randn(100),
    'B': np.random.randn(100),
    'C': np.random.randn(100)
})

# Comprehensive summary
print(df.describe())

# Individual statistics
print(f"Mean:\n{df.mean()}")
print(f"\nMedian:\n{df.median()}")
print(f"\nStd:\n{df.std()}")
print(f"\nCorrelation matrix:\n{df.corr()}")
```

## Frequency Tables

```python
# Categorical data
categories = np.random.choice(['A', 'B', 'C', 'D'], size=100)

# Frequency table
unique, counts = np.unique(categories, return_counts=True)
freq_table = dict(zip(unique, counts))
print(f"Frequency table: {freq_table}")

# Relative frequencies
rel_freq = counts / len(categories)
print(f"Relative frequencies: {dict(zip(unique, rel_freq))}")
```

## Grouped Statistics

```python
# Group by category and compute statistics
df = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C'], size=100),
    'value': np.random.randn(100)
})

grouped = df.groupby('category')['value'].agg([
    'count', 'mean', 'std', 'min', 'max'
])
print(grouped)
```
