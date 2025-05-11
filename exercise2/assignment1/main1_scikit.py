import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

#this code is the easier version of main1.py / with the help of sklearn / uni-practice2

# Load the dataset
data = load_iris()
X = data.data

# Run K-means 10 times and select the best result
best_run_losses = []
best_loss = np.inf
best_centroids = None
best_labels = None

for i in range(10):
    kmeans = KMeans(n_clusters=3, n_init=1, max_iter=300, random_state=i)
    kmeans.fit(X)

    loss = kmeans.inertia_  # Sum of squared errors (SSE)

    if loss < best_loss:
        best_loss = loss
        best_run_losses = kmeans.inertia_

        best_centroids = kmeans.cluster_centers_
        best_labels = kmeans.labels_

# Print best centroids, labels, and SSE
print("Best Centroids:")
for i, centroid in enumerate(best_centroids):
    print(f"Centroid {i}: {np.round(centroid, 4)}")

print("\nBest Labels:")
print(best_labels)

print(f"Sum of Squared Errors (SSE): {best_loss:.4f}\n")

# Plot the convergence (minimum loss per iteration)
plt.plot(best_run_losses, marker='o')
plt.title("The minimum loss among 10 iterations")
plt.xlabel("Iteration")
plt.ylabel("Sum of squared errors (SSE)")
plt.grid(True)
plt.show()
