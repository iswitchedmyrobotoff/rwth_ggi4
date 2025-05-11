import numpy as np
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris
# Load the dataset
data = load_iris()
X = data.data

# calculate distance between two points with euclid
def euclid(a, b):
    # normfunc from linaer algebra
    return np.linalg.norm(a - b, axis=1)

# K-means algorithm implementation with three randoms
def kmeans(X, k=3, max_iteration=300, upper_limit=1e-6):

    n_samples, n_features = X.shape  # assigned 150 samples and 4 features from dataset X

    # Randomly select k samples as initial centroids
    random_indices = np.random.choice(n_samples, k, replace=False)

    #assign them
    centroids = X[random_indices]

    prev_centroids = np.zeros_like(centroids)

    iteration_losses = []

    # main loop
    for iteration in range(max_iteration):

        #calculate distance between each sample point and each randomized centroid
        distances = np.array([euclid(X, centroid) for centroid in centroids])

        # assign each sample point to the closest random-selected centroid
        labels = np.argmin(distances, axis=0)

        # update the random-selected centroids (optimize them) with taking the mean of the assigned samples to them
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        #calculate the loss
        loss = sum(np.sum((X[labels == i] - new_centroids[i])**2) for i in range(k))

        iteration_losses.append(loss)

        # check if updated results nearly same as previous ones
        #if euclid(new_centroids, centroids) < upper_limit:
        #if euclid(new_centroids, centroids).sum() < upper_limit:   
        if np.linalg.norm(new_centroids - centroids) < upper_limit:
            break

        centroids = new_centroids

    return labels, centroids, loss, iteration_losses

# Run K-means 10 times and select the best result
best_loss = np.inf
best_run_losses = []
best_centroids = None
best_labels = None

for i in range(10):
    labels, centroids, loss, losses = kmeans(X, k=3)
    if loss < best_loss:
        best_loss = loss
        best_run_losses = losses
        best_centroids = centroids
        best_labels = labels

# Print best centroids, labels, and SSE


print("Best Centroids:")
for i, centroid in enumerate(best_centroids):
    print(f"Centroid {i}: {np.round(centroid, 4)}")

print("\nbest labels ")
print(best_labels)

print(f"Sum of Squared Errors (SSE): {best_loss:.4f}\n")

# Plot the convergence (minimum loss per iteration)
plt.plot(best_run_losses, marker='o')
plt.title("The minimum loss among 10 iterations")
plt.xlabel("Iteration")
plt.ylabel("Sum of squared errors (SSE)")
plt.grid(True)
plt.show()







