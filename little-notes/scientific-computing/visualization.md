# Visualization

Data visualization is essential for understanding patterns and communicating insights.

## Matplotlib Basics

```python
import matplotlib.pyplot as plt
import numpy as np

# Simple line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Wave')
plt.grid(True)
plt.show()
```

## Common Plot Types

### Line Plots
```python
x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)')
plt.legend()
plt.show()
```

### Scatter Plots
```python
x = np.random.randn(100)
y = 2*x + np.random.randn(100)*0.5

plt.scatter(x, y, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.show()
```

### Bar Charts
```python
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]

plt.bar(categories, values)
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Bar Chart')
plt.show()
```

### Histograms
```python
data = np.random.randn(1000)

plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()
```

### Heatmaps
```python
data = np.random.rand(10, 10)

plt.imshow(data, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Heatmap')
plt.show()
```

## Subplots

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sine')

# Plot 2
axes[0, 1].plot(x, np.cos(x))
axes[0, 1].set_title('Cosine')

# Plot 3
axes[1, 0].scatter(np.random.randn(50), np.random.randn(50))
axes[1, 0].set_title('Scatter')

# Plot 4
axes[1, 1].hist(np.random.randn(1000), bins=30)
axes[1, 1].set_title('Histogram')

plt.tight_layout()
plt.show()
```

## Seaborn for Statistical Plots

```python
import seaborn as sns

# Set style
sns.set_style('whitegrid')

# Distribution plot
data = np.random.randn(1000)
sns.histplot(data, kde=True)
plt.show()

# Box plot
tips = sns.load_dataset('tips')
sns.boxplot(x='day', y='total_bill', data=tips)
plt.show()

# Correlation heatmap
corr = tips.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
```
