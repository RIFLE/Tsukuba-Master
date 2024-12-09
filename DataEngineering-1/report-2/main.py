import numpy as np

####################
### ASSIGNMENT 1 ###
####################

# Distance matrix
distance_matrix = np.array([
    [0.00, 7.59, 8.11, 3.01, 1.00, 9.02, 1.00],
    [7.59, 0.00, 1.10, 6.33, 7.50, 1.20, 8.22],
    [8.11, 1.10, 0.00, 6.33, 8.34, 1.51, 7.08],
    [3.01, 6.33, 6.33, 0.00, 3.00, 9.63, 3.00],
    [1.00, 7.50, 8.34, 3.00, 0.00, 8.42, 1.00],
    [9.02, 1.20, 1.51, 9.63, 8.42, 0.00, 8.33],
    [1.00, 8.22, 7.08, 3.00, 1.00, 8.33, 0.00],
])

Eps = 1.2
MinPts = 3
n_points = distance_matrix.shape[0]

# Initialize labels (-1 means unvisited)
labels = np.full(n_points, -1)

# List to keep track of core points
core_points = []

# Step 1: Find core points
for i in range(n_points):
    neighbors = np.where(distance_matrix[i] <= Eps)[0]
    if len(neighbors) >= MinPts:
        core_points.append(i)

# Step 2: Cluster assignment
cluster_id = 0
for core_point in core_points:
    if labels[core_point] == -1:
        # Start a new cluster
        labels[core_point] = cluster_id
        # Initialize the list of points to visit
        points_to_visit = [core_point]
        while points_to_visit:
            current_point = points_to_visit.pop()
            neighbors = np.where(distance_matrix[current_point] <= Eps)[0]
            for neighbor in neighbors:
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
                    if neighbor in core_points:
                        points_to_visit.append(neighbor)
        cluster_id += 1

# Map indices to point names
point_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
clusters = {}
noise = []
for idx, label in enumerate(labels):
    if label != -1:
        cluster = clusters.get(label, [])
        cluster.append(point_names[idx])
        clusters[label] = cluster
    else:
        noise.append(point_names[idx])

# Print results
print("Core Points:", [point_names[i] for i in core_points])
print("\nClusters:")
for cluster_id, members in clusters.items():
    print(f"Cluster {cluster_id + 1}: {', '.join(members)}")
print("\nNoise Points:", ', '.join(noise))


####################
### ASSIGNMENT 2 ###
####################

print("\n")

# Data points
data = {
    'x1': np.array([35, 90]),
    'x2': np.array([78, 100]),
    'x3': np.array([80, 80]),
    'x4': np.array([79, 78]),
    'x5': np.array([91, 80]),
    'x6': np.array([20, 80]),
    'x7': np.array([52, 60]),
    'x8': np.array([105, 50])
}

# Clusters
cluster1 = ['x1', 'x6', 'x7']
cluster2 = ['x2', 'x3', 'x4', 'x5', 'x8']

### Prototype-Based Cohesion

# Function to compute centroid
def compute_centroid(cluster_points):
    return np.mean(cluster_points, axis=0)

# Function to compute cohesion
def compute_cohesion(cluster_points, centroid):
    distances = np.linalg.norm(cluster_points - centroid, axis=1)
    squared_distances = distances ** 2
    return np.sum(squared_distances)

# Cluster 1
cluster1_points = np.array([data[pt] for pt in cluster1])
centroid1 = compute_centroid(cluster1_points)
cohesion1 = compute_cohesion(cluster1_points, centroid1)

# Cluster 2
cluster2_points = np.array([data[pt] for pt in cluster2])
centroid2 = compute_centroid(cluster2_points)
cohesion2 = compute_cohesion(cluster2_points, centroid2)

# Total Cohesion
total_cohesion = cohesion1 + cohesion2

# Round to second decimal place
total_cohesion = round(total_cohesion, 2)

print(f"Prototype-Based Cohesion: {total_cohesion}")

### Prototype-Based Separation

# Compute squared Euclidean distance between centroids
centroid_distance = np.linalg.norm(centroid1 - centroid2)
separation = centroid_distance ** 2

# Round to second decimal place
separation = round(separation, 2)

print(f"Prototype-Based Separation: {separation}")

### Normalized Mutual Information (NMI)

# Data for contingency table, n_ij counts
n11 = 2  # Label 1, Cluster 1
n12 = 1  # Label 1, Cluster 2
n21 = 1  # Label 2, Cluster 1
n22 = 4  # Label 2, Cluster 2

n1_dot = n11 + n12  # Total for Label 1
n2_dot = n21 + n22  # Total for Label 2
n_dot1 = n11 + n21  # Total for Cluster 1
n_dot2 = n12 + n22  # Total for Cluster 2
N = n1_dot + n2_dot  # Total number of points

# Calculate MI
import math

def log2(x):
    return math.log(x) / math.log(2)

def mutual_information():
    MI = 0
    # Terms where n_ij > 0
    n_ij_list = [(n11, n1_dot, n_dot1), (n12, n1_dot, n_dot2),
                 (n21, n2_dot, n_dot1), (n22, n2_dot, n_dot2)]
    for n_ij, n_i_dot, n_dot_j in n_ij_list:
        if n_ij > 0:
            MI += (n_ij / N) * log2((N * n_ij) / (n_i_dot * n_dot_j))
    return MI

MI = mutual_information()

# Calculate entropies
def entropy(n_i_dot):
    p = n_i_dot / N
    return -p * log2(p)

H_L = entropy(n1_dot) + entropy(n2_dot)
H_C = entropy(n_dot1) + entropy(n_dot2)

# Calculate NMI
NMI = (2 * MI) / (H_L + H_C)

# Round to third decimal place
NMI = round(NMI, 3)

print(f"NMI: {NMI}")
