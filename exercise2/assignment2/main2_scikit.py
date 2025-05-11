import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

#this code is the easier version of main2.py / with the help of sklearn / uni-practice2

# had to include
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'car.jpg')


image = plt.imread(image_path)

#check if the image is RGB
if image.shape[2] != 3:
    raise ValueError("That was not a RGB image")

image = image / 255.0
W, H, C = image.shape
# reshape the image to a 2D array (black-gray scale)
X = image.reshape(-1, 3)  # (W*H, 3)


def compress_image_with_kmeans(X, K, max_iter=1000):
    kmeans = KMeans(n_clusters=K, init='random', max_iter=max_iter, random_state=0)
    kmeans.fit(X)
    
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    compressed_X = centroids[labels]
    
    loss = np.sum((X - centroids[labels]) ** 2)
    
    return compressed_X, loss, kmeans.inertia_


def plot_results(original_image, compressed_image, loss, K):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    
    
    axs[1].imshow(compressed_image)
    axs[1].set_title(f"Compressed Image (K={K})")
    axs[1].axis("off")
    
    plt.figure()
    plt.plot(loss, marker='o')
    plt.title(f"Loss vs Iteration (K={K})")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def main():
    losses = []  
    for K in [2, 4, 32]:
        print(f"\nProcessing K={K}, Patience is a virtue")
        
        compressed_X, loss, inertia = compress_image_with_kmeans(X, K)
        
        
        compressed_image = compressed_X.reshape(W, H, 3)
        
        
        losses.append(loss)

    
        plot_results(image, compressed_image, losses, K)

if __name__ == "__main__":
    main()
