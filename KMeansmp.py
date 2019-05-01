
import numpy as np
import random as rd
import math
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


"""
    Implementation of the Iterative KMeans-+ algorithm proposed by Hassan Ismkhan following the scikit-learn API
"""
class KMeansmp(BaseEstimator):

    def __compute_distances(self, A, B):
        distances = np.zeros((len(B), len(A)))
        for ai, a in enumerate(A):
            distances[:, ai] = np.sqrt(np.sum((B - a) ** 2, axis=1))
        return distances

    # Updating centroids for k-means
    def __update_centroids(self, D, clusters, centroids):
        for ci, centroid in enumerate(centroids):
            mask = np.array(clusters == ci)
            elems = np.sum(mask)
            if elems > 0:
                centroids[ci] = np.sum(D[mask], axis=0)/elems

    # Cost is defined as the distance squared between element in cluster and its centroid
    def __evaluate_cost(self, distances, clusters):
        SSEDM = np.zeros(self.n_clusters)
        min_distance = np.min(distances,axis=1)
        squared_min_distance = min_distance**2
        for ci in range(self.n_clusters):
            SSEDM[ci] += np.sum(squared_min_distance*(clusters == ci))
        return SSEDM

    def __centroids_sequential_init(self, D):
        centroids = np.array([D[rd.randint(0, len(D) - 1)]])  # First centroid random
        distances = self.__compute_distances(np.array(centroids), D)
        min_dist_sq = np.min(distances, axis=1) ** 2
        for k in range(self.n_clusters - 1):
            new_centroid = D[np.random.choice(len(D), 1, p=(min_dist_sq / np.sum(min_dist_sq)))]
            centroids = np.append(centroids, new_centroid, axis=0)
            distances = self.__compute_distances(centroids, D)
            min_dist_sq = np.min(distances, axis=1) ** 2
        return centroids

    def __kmeans(self, D,update=False):
        min_cost, best_clustering, best_centroids, best_distances, best_SSEDM = -1, None, None, 0, 0

        for r in range(self.n_init):

            # Initializing centroids
            if self.initialization == 'k-means++':
                centroids = self.__centroids_sequential_init(D)  # KMeans++ centroid initialization
            elif update:
                centroids = self.centroids_aux
            else:
                centroids = D[rd.sample(list(range(D.shape[0])), self.n_clusters)]  # Kmeans centroid initialization

            cost, early_stop, clusters = -1, False, None

            # Compute distance from each point to each centroid
            distances = self.__compute_distances(centroids, D)
            clusters = np.argmin(distances, axis=1)

            for i in range(self.max_iter):

                # Update centroids
                self.__update_centroids(D, clusters, centroids)

                # Compute distance from each point to each centroid
                distances = self.__compute_distances(centroids, D)
                clusters = np.argmin(distances, axis=1)

                # Evaluate current cost
                SSEDM = self.__evaluate_cost(distances, clusters)
                if math.isclose(sum(SSEDM), cost, rel_tol=self.tol):
                    break
                cost = sum(SSEDM)

            if cost < min_cost or min_cost == -1:
                min_cost = cost
                best_clustering = clusters
                best_centroids = centroids
                best_distances = distances
                best_SSEDM = SSEDM

        return best_clustering, best_centroids, best_SSEDM, best_distances




    # Iterative K-Means-+ functions

    def __computeGainCost(self, SSEDM, distances):
        # Gain:
        self.GainCost[:, 0] = SSEDM*0.75  # aplha coefficient = 3/4
        # Cost:
        distance_2 = [x[self.labels_2[i]] for i, x in enumerate(distances)]
        SSEDM_2 = np.array(distance_2) ** 2
        for ci in range(self.n_clusters):
             self.GainCost[ci, 1] = np.sum(SSEDM_2 * (self.labels_ == ci)) - SSEDM[ci]

    def __computeAdjacentClusters(self):
        for ci in range(self.n_clusters):
            self.Adjacent[ci][np.unique(self.labels_2[self.labels_ == ci])] = 1


    def __plot_solution(self,X):
         plt.scatter(X[:, 0], X[:, 1], c=self.labels_.astype(float), s=20)
         plt.scatter(self.centroids_[:,0],self.centroids_[:,1],c='r',s=80)
         plt.show()


    """
        PARAMETERS:
        n_clusters = # of clusters to be found by the algorithm
        algorithm = {'kmeans++','random'}
        n_init = # Of times the KMeansmp will run with different centroids
        max_iter = Maximum number of iterations to run for one centroid initializaion
        tol = Error toleration to accept convergence
    """
    def __init__(self, n_clusters=8, initialization='k-means++', n_init=10, max_iter=300, tol=0.0001):

        # Data structures for k-means
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.initialization = initialization

        # Data structures for the i-k-means-+
        self.GainCost = np.zeros((self.n_clusters, 2), dtype=np.float32)
        self.DivRem = np.ones((self.n_clusters, 2), dtype=np.bool)
        self.Matchable = np.ones((self.n_clusters, self.n_clusters), dtype=np.bool)
        np.fill_diagonal(self.Matchable, False)
        self.Adjacent = np.zeros((self.n_clusters, self.n_clusters), dtype=np.bool)

    def fit(self, X):
        # Step1: Apply K-Means and obtain clusters and centroids
        self.labels_, self.centroids_, SSEDM, distances = self.__kmeans(X)

        self.__plot_solution(X)

        success = 0
        while success <= self.n_clusters/2:
            # Compute second nearest center
            idx = np.argsort(distances, axis=1)
            self.labels_2 = idx[:,1]

            # Computation of cost and gain of each cluster
            self.__computeGainCost(SSEDM, distances)
            print(self.GainCost)

            # Update adjacent matrix
            self.__computeAdjacentClusters()

            # Instruction 3: Select the cluster with the largest values of gain
            divisible = self.DivRem[:, 1]
            valid_clusters = np.where(divisible)[0]
            Si = valid_clusters[np.argsort(self.GainCost[:, 0][divisible])[-1]]
            print("Cluster Si:",Si)

            # Instruction 4: If there are k/2 clusters that have gain larger than Si goto end
            gainSi = self.GainCost[Si, 0]
            larger = np.sum(self.GainCost[:, 0] > gainSi)
            if larger >= self.n_clusters/2:
                break

            # Instruction 5: Among clusters with following conditions, select a cluster S j with the smallest value
            # of Cost
            valid = self.GainCost[Si, 0] > self.GainCost[:, 1]  # Should have cost higher
            valid = np.logical_and(valid, self.DivRem[:, 1])  # Cluster Sj not irremovable
            valid = np.logical_and(valid, self.Matchable[Si, :])  # Should be matchable
            valid = np.logical_and(valid, np.logical_not(self.Adjacent[Si, :]))  # Si not adjacent to Sj
            valid = np.logical_and(valid, np.logical_not(self.Adjacent[:, Si]))  # Sj not adjacent to Si

            if not np.any(valid):
                break

            valid_clusters = np.where(valid)[0]
            Sj = valid_clusters[np.argsort(self.GainCost[:, 1][valid])[0]]
            print("Cluster Sj:", Sj)

            # Instruction 6: If there are k / 2 clusters have cost smaller than Sj
            # mark S i as an indivisible cluster and go to Instruction #3
            smaller = np.sum(self.GainCost[:, 1] < self.GainCost[Sj, 1])
            if smaller >= self.n_clusters/2:
                self.DivRem[Si, 0] = False
                continue


            # Instruction 7: Save current solution
            #                Change centroid coordinate to random point in Si,
            #                Update with k-means

            self.labels_aux = np.copy(self.labels_)
            self.centroids_aux = np.copy(self.centroids_)

            new_coordinates = X[self.labels_aux == Si][np.random.randint(0, sum(self.labels_aux == Si))]
            self.centroids_aux[Sj, :] = new_coordinates
            self.labels_aux, self.centroids_aux, SSEDM_new, distances_aux = self.__kmeans(X, update=True)

            # Instruction 8
            if np.sum(SSEDM_new) >= np.sum(SSEDM):
                self.Matchable[Si][Sj] = False
            else:
                self.DivRem[Si, 1] = False
                self.DivRem[Sj, 1] = False
                # Mark previous strong adjacent clusters of Sj as indivisible
                strong_adjacent_Sj = [ci for ci in range(self.n_clusters) if self.Adjacent[Sj, ci] and self.Adjacent[ci, Sj]]
                self.DivRem[strong_adjacent_Sj, 0] = False
                # Save new solution
                self.labels_ = np.copy(self.labels_aux)
                self.centroids_ = np.copy(self.centroids_aux)
                # Mark the current strong adjacent clusters of Si and Sj as irremovable clusters
                self.__computeAdjacentClusters()
                strong_adjacent_Sj = [ci for ci in range(self.n_clusters) if self.Adjacent[Sj, ci] and self.Adjacent[ci, Sj]]
                strong_adjacent_Si = [ci for ci in range(self.n_clusters) if self.Adjacent[Si, ci] and self.Adjacent[ci, Si]]
                self.DivRem[strong_adjacent_Sj+strong_adjacent_Si, 1] = False
                success += 1

        self.is_fitted_ = True
        return self



    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        #Code:

        return np.ones(X.shape[0], dtype=np.int64)



    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"n_clusters": self.n_clusters, "n_init": self.n_init, "max_iter": self.max_iter,
                "tol": self.tol, "initialization": self.initialization}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
