import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# 1. Making dataset
X, y = make_blobs(n_samples=500, n_features=2, centers=3, random_state=23)

# 2. Feature Scaling using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Initializing random centroids
k = 3
clusters = {}
np.random.seed(23)

for idx in range(k):
    # Randomly initialize between -2 and 2
    center = 2 * (2 * np.random.random((X.shape[1],)) - 1)
    cluster = {
        'center': center,
        'points': []
    }
    clusters[idx] = cluster

# 4. Defining Euclidean distance function
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# 5. Assigning clusters to each point
def assign_clusters(X, clusters):
    for idx in range(X.shape[0]):
        dist = []
        curr_x = X[idx]
        
        for i in range(k):
            dis = distance(curr_x, clusters[i]['center'])
            dist.append(dis)
            
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)
    return clusters

# 6. Updating the centroids based on the mean
def update_clusters(X, clusters):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis=0)
            clusters[i]['center'] = new_center
            
            # Clear the points list for the next iteration!
            clusters[i]['points'] = []
    return clusters

# 7. Predicting final clusters for plotting
def pred_cluster(X, clusters):
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i], clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred

# ==========================================
# 8. THE MISSING PIECE: Iteration & Threshold
# ==========================================
max_iters = 100
threshold = 1e-4

for i in range(max_iters):
    # Save the current centers before we update them
    old_centers = [clusters[c]['center'] for c in range(k)]
    
    # Assign and Update
    clusters = assign_clusters(X, clusters)
    clusters = update_clusters(X, clusters)
    
    # Check for convergence (the threshold logic)
    converged = True
    for c in range(k):
        new_center = clusters[c]['center']
        old_center = old_centers[c]
        
        shift = distance(new_center, old_center)
        
        # If any centroid moved more than the threshold, we haven't converged yet
        if shift > threshold:
            converged = False
            break
            
    if converged:
        print(f"Algorithm converged! Centroids stabilized after {i + 1} iterations.")
        break

# 9. Plotting the final stabilized clusters
pred = pred_cluster(X, clusters)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=pred, cmap='viridis', alpha=0.6)

for i in clusters:
    center = clusters[i]['center']
    # Plotting the final centroids as large red stars
    plt.scatter(center[0], center[1], marker='*', c='red', s=250, edgecolors='black')
    
plt.title("Final K-Means Clusters")
plt.grid(True)
plt.show()