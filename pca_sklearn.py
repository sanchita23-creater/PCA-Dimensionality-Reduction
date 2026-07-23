import numpy as np
from sklearn.decomposition import PCA

# Dataset
X = np.array([
    [2, 1],
    [4, 3],
    [6, 5],
    [8, 7]
])

print("Original Dataset:")
print(X)

# Apply PCA
pca = PCA(n_components=1)

X_pca = pca.fit_transform(X)

print("\nDataset after PCA:")
print(X_pca)

print("\nPrincipal Component:")
print(pca.components_)

print("\nExplained Variance:")
print(pca.explained_variance_)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nMean:")
print(pca.mean_)