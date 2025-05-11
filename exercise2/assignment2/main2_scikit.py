import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# Resmi yükle
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'car.jpg')

image = plt.imread(image_path)

# RGB değilse hata ver
if image.shape[2] != 3:
    raise ValueError("That was not a RGB image")

# Normalize et (0-255 -> 0-1 arası)
image = image / 255.0

# Şekli al
W, H, C = image.shape
X = image.reshape(-1, 3)

def compress_image_with_kmeans(X, K):
    kmeans = KMeans(n_clusters=K, init='random', max_iter=100, random_state=0, n_init=1)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    compressed_X = centroids[labels]
    return compressed_X, kmeans.inertia_

def run_compression(image, Ks):
    W, H, _ = image.shape
    X = image.reshape(-1, 3)

    for K in Ks:
        print(f"\nProcessing K={K}...")
        compressed_X, loss = compress_image_with_kmeans(X, K)
        compressed_image = compressed_X.reshape(W, H, 3)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(compressed_image)
        axs[1].set_title(f"Compressed Image (K={K})")
        axs[1].axis("off")

        plt.suptitle(f"Loss (Inertia): {loss:.2f}", fontsize=12)
        plt.show()

if __name__ == "__main__":
    run_compression(image, Ks=[2, 4, 32])
