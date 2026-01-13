# Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data without explicit supervision.

## Key Techniques

### Clustering
Grouping similar data points together:
- K-Means
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Models

### Dimensionality Reduction
Reducing the number of features while preserving information:
- Principal Component Analysis (PCA)
- t-SNE
- UMAP
- Autoencoders

## Example: K-Means Clustering

```python
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate sample data
X = np.random.randn(300, 2)
X[:100] += [2, 2]
X[100:200] += [-2, -2]

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=300, c='red', marker='X')
plt.title('K-Means Clustering')
plt.show()
```

## Applications

- Customer segmentation
- Image compression
- Anomaly detection
- Feature extraction
