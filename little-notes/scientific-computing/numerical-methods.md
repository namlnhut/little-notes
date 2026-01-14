# Numerical Methods

Numerical methods provide computational techniques for solving mathematical problems.

## Root Finding

### Bisection Method
```python
def bisection(f, a, b, tol=1e-6):
    """Find root of f(x) in interval [a, b]"""
    while (b - a) / 2 > tol:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

# Example: Find root of x^2 - 2
f = lambda x: x**2 - 2
root = bisection(f, 0, 2)
print(f"Square root of 2: {root}")
```

### Newton's Method
```python
def newton(f, df, x0, tol=1e-6, max_iter=100):
    """Newton's method for root finding"""
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        x = x - fx / df(x)
    return x

# Example
f = lambda x: x**2 - 2
df = lambda x: 2*x
root = newton(f, df, 1.0)
print(f"Square root of 2: {root}")
```

## Numerical Integration

### Trapezoidal Rule
```python
import numpy as np

def trapezoidal(f, a, b, n):
    """Integrate f from a to b using n trapezoids"""
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    return h * (0.5*y[0] + np.sum(y[1:-1]) + 0.5*y[-1])

# Example: Integrate sin(x) from 0 to pi
f = lambda x: np.sin(x)
result = trapezoidal(f, 0, np.pi, 100)
print(f"Integral: {result}")  # Should be close to 2
```

### Simpson's Rule
```python
def simpson(f, a, b, n):
    """Simpson's rule (n must be even)"""
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    return h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]) + y[-1])

result = simpson(np.sin, 0, np.pi, 100)
print(f"Integral: {result}")
```

## Solving Differential Equations

### Euler's Method
```python
def euler(f, y0, t):
    """Solve dy/dt = f(t, y) using Euler's method"""
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        y[i+1] = y[i] + dt * f(t[i], y[i])
    return y

# Example: dy/dt = -y, y(0) = 1
f = lambda t, y: -y
t = np.linspace(0, 5, 100)
y = euler(f, 1.0, t)

import matplotlib.pyplot as plt
plt.plot(t, y, label='Numerical')
plt.plot(t, np.exp(-t), label='Exact')
plt.legend()
plt.show()
```

### Runge-Kutta Method (RK4)
```python
def rk4(f, y0, t):
    """4th order Runge-Kutta method"""
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        k1 = f(t[i], y[i])
        k2 = f(t[i] + dt/2, y[i] + dt*k1/2)
        k3 = f(t[i] + dt/2, y[i] + dt*k2/2)
        k4 = f(t[i] + dt, y[i] + dt*k3)
        y[i+1] = y[i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y
```

## Using SciPy

```python
from scipy import integrate, optimize

# Root finding
result = optimize.root_scalar(lambda x: x**2 - 2, bracket=[0, 2])
print(f"Root: {result.root}")

# Integration
result, error = integrate.quad(np.sin, 0, np.pi)
print(f"Integral: {result}")

# Solving ODEs
def dydt(t, y):
    return -y

sol = integrate.solve_ivp(dydt, [0, 5], [1.0], dense_output=True)
t = np.linspace(0, 5, 100)
y = sol.sol(t)
```
