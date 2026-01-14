# Supervised Learning

Supervised learning involves training a model on labeled data to make predictions on unseen data.

## Key Concepts

### Classification
Classification algorithms predict discrete class labels:
- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)

### Regression
Regression algorithms predict continuous values:
- Linear Regression
- Polynomial Regression
- Ridge and Lasso Regression

## Example: Linear Regression

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate sample data
X = np.random.rand(100, 1) * 10
y = 2.5 * X + np.random.randn(100, 1) * 2

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"R² Score: {score:.3f}")
```

## Further Reading

- [Scikit-learn Documentation](https://scikit-learn.org/)
- Pattern Recognition and Machine Learning by Bishop
