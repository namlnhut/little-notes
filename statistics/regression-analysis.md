# Regression Analysis

Regression analysis models the relationship between variables.

## Simple Linear Regression

### Model
$$y = \beta_0 + \beta_1 x + \epsilon$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate data
np.random.seed(42)
x = np.random.randn(100)
y = 2 + 3*x + np.random.randn(100)

# Perform regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f"Slope: {slope:.3f}")
print(f"Intercept: {intercept:.3f}")
print(f"R²: {r_value**2:.3f}")
print(f"p-value: {p_value:.6f}")

# Plot
plt.scatter(x, y, alpha=0.5)
plt.plot(x, intercept + slope*x, 'r', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Simple Linear Regression')
plt.show()
```

### Using scikit-learn
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

X = x.reshape(-1, 1)
y_array = y

model = LinearRegression()
model.fit(X, y_array)

y_pred = model.predict(X)

print(f"Coefficient: {model.coef_[0]:.3f}")
print(f"Intercept: {model.intercept_:.3f}")
print(f"R²: {r2_score(y_array, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_array, y_pred)):.3f}")
```

## Multiple Linear Regression

### Model
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p + \epsilon$$

```python
from sklearn.linear_model import LinearRegression

# Generate data
n = 100
X = np.random.randn(n, 3)
true_coef = np.array([2, -1, 0.5])
y = 5 + X @ true_coef + np.random.randn(n) * 0.5

# Fit model
model = LinearRegression()
model.fit(X, y)

print(f"True coefficients: {true_coef}")
print(f"Estimated coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.3f}")
print(f"R²: {model.score(X, y):.3f}")
```

### Using statsmodels (for detailed statistics)
```python
import pandas as pd
import statsmodels.api as sm

# Create DataFrame
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
df['y'] = y

# Add constant for intercept
X_with_const = sm.add_constant(df[['x1', 'x2', 'x3']])

# Fit model
model = sm.OLS(df['y'], X_with_const).fit()

# Print summary
print(model.summary())
```

## Model Diagnostics

### Residual Analysis
```python
# Fit model
X = np.random.randn(100, 2)
y = 2 + 3*X[:, 0] - 1*X[:, 1] + np.random.randn(100)*0.5

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Residuals
residuals = y - y_pred

# Plot residuals
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Residuals vs fitted
axes[0].scatter(y_pred, residuals, alpha=0.5)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Fitted values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted')

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Normal Q-Q')

# Histogram of residuals
axes[2].hist(residuals, bins=20, edgecolor='black')
axes[2].set_xlabel('Residuals')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Histogram of Residuals')

plt.tight_layout()
plt.show()
```

### Checking Assumptions
```python
# 1. Linearity: Check scatter plots and residual plots

# 2. Independence: Durbin-Watson test
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(residuals)
print(f"Durbin-Watson statistic: {dw:.3f}")  # Should be ~2

# 3. Homoscedasticity: Breusch-Pagan test
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

X_const = sm.add_constant(X)
lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(residuals, X_const)
print(f"Breusch-Pagan p-value: {lm_pvalue:.3f}")

# 4. Normality: Shapiro-Wilk test
_, p_value = stats.shapiro(residuals)
print(f"Shapiro-Wilk p-value: {p_value:.3f}")
```

## Regularization

### Ridge Regression (L2)
```python
from sklearn.linear_model import Ridge

# Ridge regression: minimize ||y - Xβ||² + α||β||²
alphas = [0.01, 0.1, 1, 10, 100]

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X, y)
    print(f"α={alpha}: R²={ridge.score(X, y):.3f}, "
          f"Coef norm={np.linalg.norm(ridge.coef_):.3f}")
```

### Lasso Regression (L1)
```python
from sklearn.linear_model import Lasso

# Lasso regression: minimize ||y - Xβ||² + α||β||₁
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    print(f"α={alpha}: R²={lasso.score(X, y):.3f}, "
          f"Coef: {lasso.coef_}")
```

### Elastic Net
```python
from sklearn.linear_model import ElasticNet

# Elastic Net: combination of L1 and L2
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X, y)
print(f"R²: {elastic.score(X, y):.3f}")
print(f"Coefficients: {elastic.coef_}")
```

## Feature Selection

### Forward Selection
```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

# Generate data with some irrelevant features
X = np.random.randn(100, 10)
y = X[:, 0] + 2*X[:, 1] + np.random.randn(100)*0.1

# Forward selection
sfs = SequentialFeatureSelector(LinearRegression(),
                                 n_features_to_select=3,
                                 direction='forward')
sfs.fit(X, y)

selected_features = sfs.get_support()
print(f"Selected features: {np.where(selected_features)[0]}")
```

### Using p-values
```python
import statsmodels.api as sm

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()

# Extract p-values
p_values = model.pvalues[1:]  # Exclude intercept
significant = p_values < 0.05

print(f"Significant features (p < 0.05): {np.where(significant)[0]}")
```

## Polynomial Regression

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate non-linear data
x = np.linspace(0, 10, 100)
y = 2 + 3*x - 0.5*x**2 + np.random.randn(100)*2

X = x.reshape(-1, 1)

# Fit polynomial regression
degree = 2
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_model.fit(X, y)

# Plot
x_plot = np.linspace(0, 10, 300).reshape(-1, 1)
y_plot = poly_model.predict(x_plot)

plt.scatter(X, y, alpha=0.5)
plt.plot(x_plot, y_plot, 'r', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Polynomial Regression (degree={degree})')
plt.show()
```

## Logistic Regression

### Binary Classification
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Generate binary classification data
X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Fit model
log_reg = LogisticRegression()
log_reg.fit(X, y)

# Predictions
y_pred = log_reg.predict(X)
y_prob = log_reg.predict_proba(X)

print("Coefficients:", log_reg.coef_)
print("Intercept:", log_reg.intercept_)
print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))
print("\nClassification Report:")
print(classification_report(y, y_pred))
```

### Visualize Decision Boundary
```python
# Create mesh
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict on mesh
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()
```

## Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

# Generate data
X = np.random.randn(100, 5)
y = X[:, 0] + 2*X[:, 1] + np.random.randn(100)*0.5

# K-fold cross-validation
model = LinearRegression()
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

print(f"Cross-validation R² scores: {scores}")
print(f"Mean R²: {scores.mean():.3f} (+/- {scores.std():.3f})")
```
