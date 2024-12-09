####################
### ASSIGNMENT 1 ###
####################
import networkx as nx

# Define the graph edges
edges = [
    (1, 2), (1, 3), (2, 3),
    (3, 5), (3, 7), (4, 5),
    (4, 6), (5, 6), (5, 7),
    (7, 8), (7, 9), (8, 9)
]

# Create the graph
G = nx.Graph()
G.add_edges_from(edges)

# Define the clusters
C1 = {1, 2, 3, 4, 5, 6}
C2 = {7, 8, 9}
clusters = [C1, C2]

# Calculate Modularity Q
def calculate_modularity(G, clusters):
    m = G.number_of_edges()
    Q = 0.0
    cluster_labels = {}
    for idx, cluster in enumerate(clusters):
        for node in cluster:
            cluster_labels[node] = idx
    for u, v in G.edges():
        delta = 1 if cluster_labels[u] == cluster_labels[v] else 0
        k_u = G.degree(u)
        k_v = G.degree(v)
        Q += (1 - (k_u * k_v) / (2 * m)) * delta
    Q /= (2 * m)
    return Q

Q = calculate_modularity(G, clusters)
Q = round(Q, 3)
print(f"Modularity Q: {Q}")

####################
### ASSIGNMENT 2 ###
####################

from collections import deque
import math

# Define the graph edges
edges = [
    (1, 2), (1, 3), (2, 3),
    (3, 5), (3, 7), (4, 5),
    (4, 6), (5, 6), (5, 7),
    (7, 8), (7, 9), (8, 9)
]

# Create the graph
G = nx.Graph()
G.add_edges_from(edges)

# SCAN algorithm parameters
epsilon = 0.8
mu = 2

# Step 1: Compute Structural Similarity between nodes
def structural_similarity(G, u, v):
    neighbors_u = set(G.neighbors(u)) | {u}
    neighbors_v = set(G.neighbors(v)) | {v}
    intersection_size = len(neighbors_u & neighbors_v)
    denom = math.sqrt(len(neighbors_u) * len(neighbors_v))
    sim = intersection_size / denom if denom != 0 else 0
    return round(sim, 3)

# Compute structural similarities for all connected node pairs
SS = {}
for u, v in G.edges():
    sim = structural_similarity(G, u, v)
    SS[(u, v)] = sim
    SS[(v, u)] = sim  # Since the graph is undirected

# Step 2: Identify Core Nodes and Clusters
nodes = list(G.nodes())
cluster_labels = {}
cluster_id = 0
visited = set()

def get_eps_neighbors(u):
    return [v for v in G.neighbors(u) if SS.get((u, v), 0) >= epsilon]

for u in nodes:
    if u in visited:
        continue
    N_eps = get_eps_neighbors(u)
    if len(N_eps) >= mu - 1:
        # Node u is a core node
        cluster_id += 1
        cluster_queue = deque()
        cluster_queue.append(u)
        cluster_nodes = set()
        visited.add(u)
        while cluster_queue:
            current = cluster_queue.popleft()
            cluster_nodes.add(current)
            N_eps_current = get_eps_neighbors(current)
            if len(N_eps_current) >= mu - 1:
                for w in N_eps_current:
                    if w not in visited:
                        visited.add(w)
                        cluster_queue.append(w)
        for node in cluster_nodes:
            cluster_labels[node] = f'Cluster {cluster_id}'
    else:
        visited.add(u)

# Step 3: Identify Hubs and Outliers
unclassified_nodes = set(G.nodes()) - set(cluster_labels.keys())
hubs = set()
outliers = set()

for u in unclassified_nodes:
    neighbor_clusters = set()
    for v in G.neighbors(u):
        if v in cluster_labels:
            neighbor_clusters.add(cluster_labels[v])
    if len(neighbor_clusters) > 1:
        hubs.add(u)
    elif len(neighbor_clusters) == 0:
        outliers.add(u)
    else:
        cluster_labels[u] = neighbor_clusters.pop()

# Output the results
print("\nClusters:")
clusters = {}
for node, label in cluster_labels.items():
    clusters.setdefault(label, []).append(node)

for label, members in clusters.items():
    print(f"{label}: {sorted(members)}")

print("\nHubs:")
print(sorted(hubs))

print("\nOutliers:")
print(sorted(outliers))
