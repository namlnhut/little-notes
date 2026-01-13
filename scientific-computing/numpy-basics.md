# NumPy Basics

NumPy is the fundamental package for numerical computing in Python.

## Array Creation

```python
import numpy as np

# Create arrays
a = np.array([1, 2, 3, 4, 5])
b = np.zeros((3, 4))
c = np.ones((2, 3))
d = np.arange(0, 10, 2)
e = np.linspace(0, 1, 5)

# Random arrays
rand = np.random.rand(3, 3)
randn = np.random.randn(3, 3)
```

## Array Operations

### Element-wise Operations
```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Arithmetic
print(a + b)
print(a * b)
print(a ** 2)

# Universal functions
print(np.sin(a))
print(np.exp(a))
print(np.sqrt(a))
```

### Array Manipulation
```python
# Reshaping
arr = np.arange(12)
reshaped = arr.reshape(3, 4)

# Transposing
transposed = reshaped.T

# Stacking
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
vstacked = np.vstack([a, b])
hstacked = np.hstack([a, b])
```

## Indexing and Slicing

```python
arr = np.arange(10)

# Basic indexing
print(arr[0])
print(arr[-1])
print(arr[2:5])

# Boolean indexing
print(arr[arr > 5])

# Fancy indexing
indices = [1, 3, 5]
print(arr[indices])
```

## Broadcasting

```python
# Broadcasting allows operations on arrays of different shapes
a = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])

result = a + b  # b is broadcast to match shape of a
```

## Aggregations

```python
arr = np.random.randn(3, 4)

# Statistical functions
print(arr.sum())
print(arr.mean())
print(arr.std())
print(arr.min())
print(arr.max())

# Along axes
print(arr.sum(axis=0))  # Sum along columns
print(arr.mean(axis=1))  # Mean along rows
```
