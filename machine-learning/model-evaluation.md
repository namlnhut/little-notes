# Model Evaluation

Proper evaluation is crucial for understanding model performance and avoiding overfitting.

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed breakdown of predictions

### Regression Metrics
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**: Coefficient of determination

## Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Create model
model = RandomForestClassifier(n_estimators=100)

# Perform 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

## Bias-Variance Tradeoff

- **High Bias**: Underfitting - model is too simple
- **High Variance**: Overfitting - model is too complex
- **Goal**: Find the right balance

## Best Practices

1. Always use separate train/validation/test sets
2. Use cross-validation for robust evaluation
3. Choose metrics appropriate for your problem
4. Watch for class imbalance
5. Consider computational costs
