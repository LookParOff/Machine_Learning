import numpy as np
from collections.abc import Iterable


def distance(x, y):
    if isinstance(x, Iterable):
        return (sum([(xi - yi)**2 for xi, yi in zip(x, y)])) ** 0.5  # euclid metric space
    return abs(x - y)


def k_means(vectors: np.array, countOfClusters):
    if type(vectors[0]) != np.ndarray:
        DIM = 1
    else:
        DIM = len(vectors[0])
    centroids = np.random.random((countOfClusters, DIM)) * vectors.max()
    nearestClusters = [0 for _ in range(len(vectors))]  # nearest cluster for each vector
    while True:
        newCentroids = centroids.copy() * 0
        for index, vec in enumerate(vectors):
            distsToClusters = [distance(vec, center) for center in centroids]
            nearest = distsToClusters.index(min(distsToClusters))
            nearestClusters[index] = nearest
            newCentroids[nearest] += vectors[index]

        for i in range(countOfClusters):
            c = nearestClusters.count(i)
            if c == 0:
                newCentroids[i] = centroids[i].copy()
            else:
                newCentroids[i] /= c
        if abs((centroids - newCentroids).sum()) < 0.1*DIM:  # diff of all coord
            break
        centroids = newCentroids.copy()
    return centroids


print(k_means(np.array([[1, 1], [2, 2], [5, 5], [6, 6]]), 2))
print(k_means(np.array([1, 1.1, 1.2, 50, 50.1, 50.2]), 2))
