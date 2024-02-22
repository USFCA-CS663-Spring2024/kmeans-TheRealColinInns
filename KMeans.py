## Author: Colin Inns
import math

from cluster import Cluster
import numpy as np


def find_cluster(curr, centroids):
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
        c_index = np.random.choice(X.shape[0], size=self.k, replace=False)
        centroids = X[c_index]
        clusters = []
        counter = 0
        while counter < self.max_iter:
            clusters.clear()
            for i in range(self.k):
                clusters.append([])
            for curr in X:
                clusters[np.where(centroids == find_cluster(curr, centroids))[0][0]].append(curr)
            new_centroids = []
            for i in range(len(clusters)):
                new_centroids.append([])
                total_x = 0
                total_y = 0
                for pair in clusters[i]:
                    total_x += pair[0]
                    total_y += pair[1]
                if len(clusters[i]) > 0:
                    new_centroids[i] = ([(total_x // len(clusters[i])), (total_y // len(clusters[i]))])
                else:
                    new_centroids[i] = centroids[i]
            new_centroids = np.array(new_centroids)
            if np.array_equal(centroids, new_centroids):
                break
            else:
                centroids = new_centroids
            counter += 1
        cluster_id = []
        for item in X:
            for i in range(len(clusters)):
                for pair in clusters[i]:
                    if (item[0] == pair[0]) and (item[1] == pair[1]):
                        cluster_id.append(i)
                        break
                else:
                    continue
                break
        return cluster_id, centroids




"""
if __name__ == "__main__":
    meanie = Kmeans(folds=2)
    fitput = np.array([[0, 0], [2, 2], [0, 2], [2, 0], [10, 10], [8, 8], [10, 8], [8, 10]])
    results = meanie.fit(fitput)
    print(results[0])
    print(results[1])
"""