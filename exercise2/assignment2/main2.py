import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import os


# Load the dataset
data = load_iris()
X = data.data

# had to include this, smth was off with the path.
script_dir = os.path.dirname(os.path.abspath(__file__))  
image_path = os.path.join(script_dir, 'car.jpg')


image = plt.imread(image_path)
    #matplotlib.org
    #(M, N) for grayscale images.
    #(M, N, 3) for RGB images.
    #(M, N, 4) for RGBA images.

# check if it was a RGB iamge
if image.shape[2] != 3:
    raise ValueError("That was not a RGB image")
     
# normalization , RGB is normally = (255, 255, 255)
image = image / 255.0

# calculate distance between two points with euclid
def euclid(a, b):
    # normfunc from linaer algebra
    return np.linalg.norm(a - b, axis=1)


def initialize_centroids(X, K):
    # Randomly select K samples as initial centroids
    # with random choice from numpy return a random 1D array of K samples from X
    random_indices = np.random.choice(X.shape[0], K, replace=False)
    return X[random_indices]



def assign_clusters(X, centroids):
    # calculate distance between each sample point and each centroid
    distances = np.array([euclid(X, centroid) for centroid in centroids])
    # assign each sample point to the closest centroid
    return np.argmin(distances, axis=0)


def update_centroids(X, labels, K):
    # calculate the new updated centroids as the mean of the assigned samples to them
    return np.array([X[labels == i].mean(axis=0) for i in range(K)])


def compute_loss(X, centroids, labels):
    #sum of squared distances from each X[i] to centroids[labels[i]]
    # calculate the loss as the sum of squared distances from each sample point to its assigned centroid
    return sum(np.sum((X[labels == i] - centroids[i])**2) for i in range(len(centroids)))

def main_loop(X, K=3, max_iterations=50, upper_limit=1e-6):

    centroids = initialize_centroids(X, K)
    losses = []

    for i in range(max_iterations):
        # assign each sample point to the closest centroid
        labels = assign_clusters(X, centroids)

        loss = compute_loss(X, centroids, labels) 
        losses.append(loss)

        # update the centroids based on the assigned samples
        new_centroids = update_centroids(X, labels, K)

        if np.linalg.norm(new_centroids - centroids, axis=None) < upper_limit:
            centroids = new_centroids
            break

        centroids = new_centroids  

    return centroids, losses, labels


def image_recons(image, K):
    # take the 3D image and turn it into a 2D array
    W, H, C = image.shape
    # (W*H, 3) / turn the image to a 2D array for grayscale
    X = image.reshape(-1, 3)

    #invoke the main loop on them
    centroids, losses, labels = main_loop(X, K)

    compressed_X = centroids[labels]

    compressed_image = compressed_X.reshape(W, H, 3)

    return compressed_image, losses
    




def main():

    for K in [2, 4, 32]:
        print(f"\nProcessing K={K}, Patience is a virtue")
        compressed_image, losses = image_recons(image, K)

        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(compressed_image)
        axs[1].set_title(f"Compressed Image (K={K})")
        axs[1].axis("off")

        
        plt.figure()
        plt.plot(losses, marker='o')
        plt.title(f"Loss vs Iteration (K={K})")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.show()

if __name__ == "__main__":
    main()
