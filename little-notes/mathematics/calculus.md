# Calculus

Calculus deals with rates of change (derivatives) and accumulation (integrals).

## Derivatives

### Single Variable Calculus
The derivative represents the rate of change:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

### Common Derivatives
- $\frac{d}{dx}(x^n) = nx^{n-1}$
- $\frac{d}{dx}(e^x) = e^x$
- $\frac{d}{dx}(\ln x) = \frac{1}{x}$
- $\frac{d}{dx}(\sin x) = \cos x$
- $\frac{d}{dx}(\cos x) = -\sin x$

### Numerical Derivatives
```python
import numpy as np

def derivative(f, x, h=1e-5):
    """Numerical derivative using finite differences"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Example: derivative of x^2
f = lambda x: x**2
x = 3
df = derivative(f, x)
print(f"f'(3) = {df}")  # Should be 6
```

## Multivariate Calculus

### Partial Derivatives
```python
def partial_derivative(f, x, i, h=1e-5):
    """Partial derivative with respect to x[i]"""
    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[i] += h
    x_minus[i] -= h
    return (f(x_plus) - f(x_minus)) / (2 * h)

# Example: f(x, y) = x^2 + xy + y^2
f = lambda x: x[0]**2 + x[0]*x[1] + x[1]**2
x = np.array([2.0, 3.0])

df_dx = partial_derivative(f, x, 0)
df_dy = partial_derivative(f, x, 1)
print(f"∂f/∂x = {df_dx}")  # 2x + y = 7
print(f"∂f/∂y = {df_dy}")  # x + 2y = 8
```

### Gradient
The gradient is a vector of all partial derivatives:

```python
def gradient(f, x, h=1e-5):
    """Compute gradient of f at x"""
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        grad[i] = partial_derivative(f, x, i, h)
    return grad

# Example
grad = gradient(f, x)
print(f"∇f = {grad}")
```

### Jacobian Matrix
```python
def jacobian(f, x, h=1e-5):
    """Compute Jacobian matrix of vector function f"""
    n = len(x)
    f_val = f(x)
    m = len(f_val)
    J = np.zeros((m, n))

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        J[:, i] = (f(x_plus) - f(x_minus)) / (2 * h)

    return J

# Example: f(x, y) = [x^2 + y, xy]
f = lambda x: np.array([x[0]**2 + x[1], x[0]*x[1]])
x = np.array([2.0, 3.0])
J = jacobian(f, x)
print(f"Jacobian:\n{J}")
```

### Hessian Matrix
```python
def hessian(f, x, h=1e-4):
    """Compute Hessian matrix (second derivatives)"""
    n = len(x)
    H = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += h; x_pp[j] += h
            x_pm[i] += h; x_pm[j] -= h
            x_mp[i] -= h; x_mp[j] += h
            x_mm[i] -= h; x_mm[j] -= h

            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h**2)

    return H
```

## Integration

### Numerical Integration
```python
def integrate(f, a, b, n=1000):
    """Numerical integration using trapezoidal rule"""
    x = np.linspace(a, b, n)
    y = f(x)
    return np.trapz(y, x)

# Example: integrate x^2 from 0 to 1
f = lambda x: x**2
result = integrate(f, 0, 1)
print(f"∫x² dx from 0 to 1 = {result}")  # Should be 1/3
```

## Taylor Series

Approximate functions using polynomials:

$$f(x) \approx f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + ...$$

```python
def taylor_exp(x, n=10):
    """Taylor series approximation of e^x"""
    result = 0
    for k in range(n):
        result += x**k / np.math.factorial(k)
    return result

x = 1.0
approx = taylor_exp(x)
exact = np.exp(x)
print(f"Approximation: {approx}")
print(f"Exact: {exact}")
print(f"Error: {abs(approx - exact)}")
```

## Applications in Machine Learning

### Gradient Descent
```python
def gradient_descent(f, grad_f, x0, learning_rate=0.01, n_iter=100):
    """Minimize f using gradient descent"""
    x = x0.copy()
    history = [x.copy()]

    for i in range(n_iter):
        grad = grad_f(x)
        x = x - learning_rate * grad
        history.append(x.copy())

    return x, np.array(history)

# Example: minimize f(x, y) = x^2 + y^2
f = lambda x: x[0]**2 + x[1]**2
grad_f = lambda x: 2 * x

x0 = np.array([5.0, 5.0])
x_min, history = gradient_descent(f, grad_f, x0)
print(f"Minimum at: {x_min}")
```

### Backpropagation
Uses the chain rule to compute gradients in neural networks:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$
