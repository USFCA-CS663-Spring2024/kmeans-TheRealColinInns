## Author: Colin Inns
import math

from cluster import Cluster
import numpy as np


def find_cluster(curr, centroids): # finds the closest centroid to the current position
    min_dist = math.inf
    centroid = centroids[0]
    for c in centroids:
        dist = math.sqrt((curr[0] - c[0])**2 + (curr[1] - c[1])**2)
        if dist < min_dist:
            min_dist = dist
            centroid = c
    return centroid


class Kmeans(Cluster):
    global k
    global max_iter

    def __init__(self, folds=5, max_iterations=100):
        super().__init__()
        self.k = folds
        self.max_iter = max_iterations

    def fit(self, X):
        c_index = np.random.choice(X.shape[0], size=self.k, replace=False) # get the centroid indexes
        centroids = X[c_index] # get the centroids
        clusters = [] # keep track of which points go in which cluster
        counter = 0
        while counter < self.max_iter:
            clusters.clear() # rebuild the clusters each time
            for i in range(self.k):
                clusters.append([])
            for curr in X:
                clusters[np.where(centroids == find_cluster(curr, centroids))[0][0]].append(curr) # put the current point into the correct cluster
            new_centroids = []
            for i in range(len(clusters)): # calculate the new clusters
                new_centroids.append([])
                total_x = 0
                total_y = 0
                for pair in clusters[i]:
                    total_x += pair[0]
                    total_y += pair[1]
                if len(clusters[i]) > 0:
                    new_centroids[i] = ([(total_x / len(clusters[i])), (total_y / len(clusters[i]))])
                else:
                    new_centroids[i] = centroids[i]
            new_centroids = np.array(new_centroids)
            if np.array_equal(centroids, new_centroids):
                break
            else:
                centroids = new_centroids
            counter += 1
        cluster_id = []
        for item in X: # transform the clusters into a single array
            for i in range(len(clusters)):
                for pair in clusters[i]:
                    if (item[0] == pair[0]) and (item[1] == pair[1]):
                        cluster_id.append(i)
                        break
                else:
                    continue
                break
        return np.array(cluster_id), centroids