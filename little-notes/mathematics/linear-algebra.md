# Linear Algebra

Linear algebra is the study of vectors, matrices, and linear transformations.

## Vectors

### Vector Operations
```python
import numpy as np

# Create vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Addition and subtraction
v_sum = v1 + v2
v_diff = v1 - v2

# Scalar multiplication
v_scaled = 2 * v1

# Dot product
dot_product = np.dot(v1, v2)
print(f"Dot product: {dot_product}")

# Magnitude (L2 norm)
magnitude = np.linalg.norm(v1)
print(f"Magnitude: {magnitude}")

# Cross product (3D vectors)
cross = np.cross(v1, v2)
print(f"Cross product: {cross}")
```

## Matrices

### Matrix Operations
```python
# Create matrices
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

# Matrix addition
C = A + B

# Matrix multiplication
D = np.matmul(A, B)  # or A @ B

# Transpose
A_T = A.T

# Trace
trace = np.trace(A)

# Determinant
det = np.linalg.det(A)
```

### Special Matrices
```python
# Identity matrix
I = np.eye(3)

# Diagonal matrix
diag = np.diag([1, 2, 3])

# Zero matrix
zeros = np.zeros((3, 3))

# Random matrix
rand = np.random.randn(3, 3)
```

## Matrix Decomposition

### Eigenvalues and Eigenvectors
```python
# Eigendecomposition
A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")
```

### Singular Value Decomposition (SVD)
```python
# SVD: A = U @ S @ V^T
A = np.random.randn(5, 3)
U, s, Vt = np.linalg.svd(A)

print(f"U shape: {U.shape}")
print(f"Singular values: {s}")
print(f"V^T shape: {Vt.shape}")
```

## Linear Systems

### Solving Linear Equations
```python
# Solve Ax = b
A = np.array([[3, 1],
              [1, 2]])
b = np.array([9, 8])

x = np.linalg.solve(A, b)
print(f"Solution: {x}")

# Verify
print(f"Ax = {A @ x}")
```

### Matrix Inverse
```python
A = np.array([[1, 2],
              [3, 4]])

# Inverse
A_inv = np.linalg.inv(A)

# Verify: A @ A_inv = I
print(A @ A_inv)
```

## Applications in Machine Learning

### Principal Component Analysis (PCA)
```python
from sklearn.decomposition import PCA

# Generate sample data
X = np.random.randn(100, 5)

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance: {pca.explained_variance_ratio_}")
```

### Linear Regression (Matrix Form)
```python
# y = Xβ + ε
# Solution: β = (X^T X)^(-1) X^T y

X = np.random.randn(100, 3)
true_beta = np.array([1.5, -2.0, 0.5])
y = X @ true_beta + np.random.randn(100) * 0.1

# Solve using normal equations
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"True coefficients: {true_beta}")
print(f"Estimated coefficients: {beta_hat}")
```

## Key Concepts

- **Vector Space**: Collection of vectors with addition and scalar multiplication
- **Linear Independence**: Vectors that cannot be written as linear combinations
- **Basis**: Set of linearly independent vectors that span the space
- **Rank**: Maximum number of linearly independent rows/columns
- **Orthogonality**: Vectors with zero dot product
