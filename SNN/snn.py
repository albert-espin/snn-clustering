import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph


def snn(X, neighbor_num, min_shared_neighbor_num):
    """Perform Shared Nearest Neighbor (SNN) clustering algorithm clustering.

    Parameters
    ----------
    X : array or sparse (CSR) matrix of shape (n_samples, n_features), or array of shape (n_samples, n_samples)
    A feature array
    neighbor_num : int
    K number of neighbors to consider for shared nearest neighbor similarity
    min_shared_neighbor_num : int
    Number of nearest neighbors that need to share two data points to be considered part of the same cluster
    """

    # for each data point, find their set of K nearest neighbors
    knn_graph = kneighbors_graph(X, n_neighbors=neighbor_num, include_self=False)
    neighbors = np.array([set(knn_graph[i].nonzero()[1]) for i in range(len(X))])

    # the distance matrix is computed as the complementary of the proportion of shared neighbors between each pair of data points
    snn_distance_matrix = np.asarray([[get_snn_distance(neighbors[i], neighbors[j]) for j in range(len(neighbors))] for i in range(len(neighbors))])

    # perform DBSCAN with the shared-neighbor distance criteria for density estimation
    dbscan = DBSCAN(min_samples=min_shared_neighbor_num, metric="precomputed")
    dbscan = dbscan.fit(snn_distance_matrix)
    return dbscan.core_sample_indices_, dbscan.labels_


def get_snn_similarity(x0, x1):
    """Calculate the shared-neighbor similarity of two sets of nearest neighbors, normalized by the maximum number of shared neighbors"""

    return len(x0.intersection(x1)) / len(x0)


def get_snn_distance(x0, x1):
    """Calculate the shared-neighbor distance of two sets of nearest neighbors, normalized by the maximum number of shared neighbors"""

    return 1 - get_snn_similarity(x0, x1)


class SNN(BaseEstimator, ClusterMixin):
    """Class for performing the Shared Nearest Neighbor (SNN) clustering algorithm.

    Parameters
    ----------
    neighbor_num : int
        K number of neighbors to consider for shared nearest neighbor similarity

    min_shared_neighbor_proportion : float [0, 1]
        Proportion of the K nearest neighbors that need to share two data points to be considered part of the same cluster

    Note: Naming conventions for attributes are based on the analogous ones of DBSCAN
    """

    def __init__(self, neighbor_num, min_shared_neighbor_proportion):

        """Constructor"""

        self.neighbor_num = neighbor_num
        self.min_shared_neighbor_num = round(neighbor_num * min_shared_neighbor_proportion)

    def fit(self, X):

        """Perform SNN clustering from features or distance matrix.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or array of shape (n_samples, n_samples)
            A feature array
        """

        clusters = snn(X, neighbor_num=self.neighbor_num, min_shared_neighbor_num=self.min_shared_neighbor_num)
        self.core_sample_indices_, self.labels_ = clusters
        if len(self.core_sample_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_sample_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.
        sample_weight : array, shape (n_samples,), optional
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with negative
            weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        y : Ignored

        Returns
        -------
        y : ndarray, shape (n_samples,)
            cluster labels
        """
        self.fit(X)
        return self.labels_
