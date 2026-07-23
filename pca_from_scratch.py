import numpy as np

# Step 1: Create the dataset
X = np.array([
    [2, 1],
    [4, 3],
    [6, 5],
    [8, 7]
])

print("Original Dataset:\n", X)

# Step 2: Calculate the mean
mean = np.mean(X, axis=0)
print("\nMean of each feature:")
print(mean)

# Step 3: Center the data
X_centered = X - mean
print("\nCentered Data:")
print(X_centered)

# Step 4: Calculate the covariance matrix
cov_matrix = np.cov(X_centered.T)
print("\nCovariance Matrix:")
print(cov_matrix)

# Step 5: Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)

# Step 6: Sort eigenvalues and eigenvectors in descending order
sorted_index = np.argsort(eigenvalues)[::-1]

sorted_eigenvalues = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:, sorted_index]

print("\nSorted Eigenvalues:")
print(sorted_eigenvalues)

print("\nSorted Eigenvectors:")
print(sorted_eigenvectors)

# Step 7: Select the first principal component
principal_component = sorted_eigenvectors[:, 0]

print("\nPrincipal Component (PC1):")
print(principal_component)

# Step 8: Project the data onto the principal component
X_pca = np.dot(X_centered, principal_component)

print("\nProjected Data (1D after PCA):")
print(X_pca)