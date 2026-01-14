# Optimization

Optimization involves finding the best solution from all feasible solutions.

## Problem Formulation

General optimization problem:
$$\min_{x} f(x) \quad \text{subject to} \quad g_i(x) \leq 0, h_j(x) = 0$$

- $f(x)$: objective function
- $g_i(x)$: inequality constraints
- $h_j(x)$: equality constraints

## Unconstrained Optimization

### Gradient Descent
```python
import numpy as np

def gradient_descent(f, grad_f, x0, lr=0.01, max_iter=1000, tol=1e-6):
    """Standard gradient descent"""
    x = x0.copy()
    history = {'x': [x.copy()], 'f': [f(x)]}

    for i in range(max_iter):
        grad = grad_f(x)
        x_new = x - lr * grad

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new
        history['x'].append(x.copy())
        history['f'].append(f(x))

    return x, history

# Example: minimize f(x, y) = (x-3)^2 + (y-2)^2
f = lambda x: (x[0]-3)**2 + (x[1]-2)**2
grad_f = lambda x: 2*np.array([x[0]-3, x[1]-2])

x0 = np.array([0.0, 0.0])
x_min, hist = gradient_descent(f, grad_f, x0, lr=0.1)
print(f"Minimum at: {x_min}")
```

### Momentum
```python
def momentum_gd(f, grad_f, x0, lr=0.01, momentum=0.9, max_iter=1000):
    """Gradient descent with momentum"""
    x = x0.copy()
    v = np.zeros_like(x)
    history = [x.copy()]

    for i in range(max_iter):
        grad = grad_f(x)
        v = momentum * v - lr * grad
        x = x + v
        history.append(x.copy())

    return x, np.array(history)
```

### Adam Optimizer
```python
def adam(f, grad_f, x0, lr=0.001, beta1=0.9, beta2=0.999,
         eps=1e-8, max_iter=1000):
    """Adam optimizer"""
    x = x0.copy()
    m = np.zeros_like(x)  # First moment
    v = np.zeros_like(x)  # Second moment

    for t in range(1, max_iter + 1):
        grad = grad_f(x)

        # Update biased moments
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2

        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Update parameters
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)

    return x
```

### Newton's Method
```python
def newton_method(f, grad_f, hess_f, x0, max_iter=100):
    """Newton's method using Hessian"""
    x = x0.copy()

    for i in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)

        # Newton step
        try:
            delta = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            delta = -grad  # Fallback to gradient descent

        x = x + delta

        if np.linalg.norm(grad) < 1e-6:
            break

    return x
```

## Constrained Optimization

### Lagrange Multipliers
For equality constraints, use Lagrangian:
$$L(x, \lambda) = f(x) + \sum_i \lambda_i h_i(x)$$

```python
from scipy.optimize import minimize

# Minimize x^2 + y^2 subject to x + y = 1
def objective(x):
    return x[0]**2 + x[1]**2

def constraint(x):
    return x[0] + x[1] - 1

cons = {'type': 'eq', 'fun': constraint}
x0 = np.array([0.0, 0.0])

result = minimize(objective, x0, constraints=cons)
print(f"Minimum: {result.x}")
print(f"Value: {result.fun}")
```

## Convex Optimization

### Properties of Convex Functions
A function is convex if:
$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

```python
def is_convex_quadratic(A):
    """Check if f(x) = x^T A x is convex"""
    eigenvalues = np.linalg.eigvals(A)
    return np.all(eigenvalues >= 0)

# Example
A = np.array([[2, 0],
              [0, 2]])
print(f"Is convex: {is_convex_quadratic(A)}")
```

## Global Optimization

### Simulated Annealing
```python
def simulated_annealing(f, x0, T0=100, alpha=0.95, max_iter=1000):
    """Simulated annealing for global optimization"""
    x = x0.copy()
    f_best = f(x)
    x_best = x.copy()
    T = T0

    for i in range(max_iter):
        # Generate neighbor
        x_new = x + np.random.randn(len(x)) * 0.5
        f_new = f(x_new)

        # Accept or reject
        delta = f_new - f(x)
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            x = x_new
            if f_new < f_best:
                x_best = x_new
                f_best = f_new

        # Cool down
        T *= alpha

    return x_best
```

## Using SciPy

```python
from scipy.optimize import minimize, differential_evolution

# Unconstrained minimization
def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

x0 = np.array([0.0, 0.0])
result = minimize(rosenbrock, x0, method='BFGS')
print(f"Minimum: {result.x}")

# Global optimization
bounds = [(-5, 5), (-5, 5)]
result = differential_evolution(rosenbrock, bounds)
print(f"Global minimum: {result.x}")
```

## Applications in Machine Learning

### Linear Regression (Closed Form)
```python
# Minimize ||Xβ - y||^2
X = np.random.randn(100, 3)
y = np.random.randn(100)

# Optimal solution: β = (X^T X)^(-1) X^T y
beta = np.linalg.lstsq(X, y, rcond=None)[0]
```

### Ridge Regression
```python
# Minimize ||Xβ - y||^2 + λ||β||^2
lambda_reg = 0.1
beta_ridge = np.linalg.inv(X.T @ X + lambda_reg * np.eye(3)) @ X.T @ y
```

### Logistic Regression
```python
from scipy.optimize import minimize

def logistic_loss(beta, X, y):
    """Logistic regression loss"""
    z = X @ beta
    return np.mean(np.log(1 + np.exp(-y * z)))

# Optimize
X = np.random.randn(100, 3)
y = np.sign(np.random.randn(100))
beta0 = np.zeros(3)

result = minimize(lambda b: logistic_loss(b, X, y), beta0)
beta_opt = result.x
```
