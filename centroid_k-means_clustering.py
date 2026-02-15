import numpy as np
import pandas as pd
import tifffile as tiff
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist

# Load the binary Z-stack TIFF file
file_path = "input.tif"
image = tiff.imread(file_path)

# Label connected components (mitochondria) in the image
labeled_image = label(image)

# Get properties of labeled regions (mitochondria)
regions = regionprops(labeled_image)

# Collect centroid coordinates for each region
centroids = []
for region in regions:
    centroids.append(region.centroid)  # x, y, z centroids

# Convert centroids to a numpy array for clustering
centroids = np.array(centroids)

# Voxel size (in micrometers) for the x, y, and z axes
voxel_size = (0.1, 0.1, 0.1)  # Replace with actual voxel size in micrometers if different

# Adjust the centroids by voxel size (scale them to micrometers)
scaled_centroids = centroids * voxel_size

# Optional: Define the range of K values to test for clustering
k_values = range(1, 11)  # Cluster sizes from 1 to 10

# Function to perform KMeans clustering and compute the silhouette score
def perform_kmeans(coords, k_values):
    silhouette_scores = []

    for k in k_values:
        if k > len(coords):  # Skip if k is greater than the number of centroids
            print(f"Skipping K={k} because there are fewer centroids.")
            continue
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(coords)

        # Skip silhouette score calculation if only one cluster is found
        if len(set(kmeans.labels_)) > 1:
            score = silhouette_score(coords, kmeans.labels_)
            silhouette_scores.append((k, score))
            print(f'K={k}, Silhouette Score={score:.3f}')
        else:
            print(f"Skipping silhouette score for K={k} because only 1 cluster was found.")
    
    return silhouette_scores

# Perform clustering and get the silhouette scores for each K
silhouette_scores = perform_kmeans(scaled_centroids, k_values)

# If no valid silhouette scores are found
if not silhouette_scores:
    print("No valid silhouette scores were found.")
    exit()

# Extract the K values and corresponding silhouette scores
k_values_list = [score[0] for score in silhouette_scores]
silhouette_values = [score[1] for score in silhouette_scores]

# Plot silhouette scores for each K
plt.plot(k_values_list, silhouette_values, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different K Values')
plt.xticks(k_values_list)
plt.grid(True)
plt.show()

# Let the user choose the best K based on the silhouette score plot
best_k = int(input("Choose the best K value from the plot above: "))

# Perform KMeans clustering with the selected K
kmeans = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans.fit_predict(scaled_centroids)

# Calculate centroid locations
centroid_locations = kmeans.cluster_centers_

# Calculate the Euclidean distance between cluster centroids
distances_between_centroids = cdist(centroid_locations, centroid_locations, 'euclidean')

# Average distance between centroids
avg_centroid_distance = np.mean(distances_between_centroids[np.triu_indices(best_k, k=1)])

# Calculate the closest mitochondria neighbour distance for each mitochondrion
min_distance_to_neighbour = []
for i, point in enumerate(scaled_centroids):
    cluster_points = scaled_centroids[labels == labels[i]]
    distances = np.linalg.norm(cluster_points - point, axis=1)
    min_distance_to_neighbour.append(np.min(distances[distances > 0]))  # Ignore self-distance

# Calculate the most frequent distance between mitochondria (mode of distances)
all_distances = []
for i in range(len(scaled_centroids)):
    for j in range(i+1, len(scaled_centroids)):
        all_distances.append(np.linalg.norm(scaled_centroids[i] - scaled_centroids[j]))

most_frequent_distance = np.argmax(np.bincount(np.round(all_distances, decimals=2).astype(int)))

# Visualize the clusters, centroids, and connections between centroids
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each cluster in a different color and connect each point to its own cluster centroid
for i in range(best_k):
    cluster_points = scaled_centroids[labels == i]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {i+1}')
    
    # Draw lines connecting cluster points to their corresponding centroid
    centroid = centroid_locations[i]
    for point in cluster_points:
        ax.plot([point[0], centroid[0]], [point[1], centroid[1]], [point[2], centroid[2]], color='gray', linestyle='--')

# Plot centroids
ax.scatter(centroid_locations[:, 0], centroid_locations[:, 1], centroid_locations[:, 2], color='black', marker='X', s=200, label='Centroids')

ax.set_xlabel('X (μm)')
ax.set_ylabel('Y (μm)')
ax.set_zlabel('Z (μm)')
ax.set_title(f'Mitochondria Clustering (K={best_k})')
ax.legend()

plt.show()

# Displaying results
print(f"Average distance between cluster centroids: {avg_centroid_distance:.3f} um")
print(f"Most frequent distance between mitochondria: {most_frequent_distance} um")

# Optionally, you can display additional information about the closest neighbour distances
print(f"Average closest mitochondria neighbour distance: {np.mean(min_distance_to_neighbour):.3f} um")

# Create a new plot without axes and box
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each cluster in a different color and connect each point to its own cluster centroid
for i in range(best_k):
    cluster_points = scaled_centroids[labels == i]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {i+1}')
    
    # Draw lines connecting cluster points to their corresponding centroid
    centroid = centroid_locations[i]
    for point in cluster_points:
        ax.plot([point[0], centroid[0]], [point[1], centroid[1]], [point[2], centroid[2]], color='gray', linestyle='--')

# Plot centroids
ax.scatter(centroid_locations[:, 0], centroid_locations[:, 1], centroid_locations[:, 2], color='black', marker='X', s=200, label='Centroids')

# Remove axis and box
ax.set_axis_off()

# Set the title without axes
ax.set_title(f'Mitochondria Clustering (K={best_k}) - No Axis')

plt.show()


save_path = "output_path.png"

fig.savefig(save_path, dpi=600, bbox_inches='tight', transparent=True)