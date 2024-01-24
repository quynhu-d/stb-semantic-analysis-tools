import numpy as np
from scipy.special import gamma
from sklearn.neighbors import KDTree
from collections import defaultdict
from tqdm.notebook import tqdm
from scipy.spatial.distance import pdist, squareform, euclidean
# from FuzzyNew import Fuzzifier, fuzzy_dist
import time


class Wishart_fuzzy:
    def __init__(self, wishart_neighbors, significance_level, dim, l=None, r=None, dc=None):
        self.wishart_neighbors = wishart_neighbors  # Number of neighbors
        self.significance_level = significance_level  # Significance level
        self.l = l  # length of left slope
        self.r = r  # length of right slope
        self.dc = dc  # length of mu == 1
        self.dim = dim

    def fit(self, X, mus=None, fuzzified=False, precomputed=False, verbose=False):
        """
            X (n_objects, n_features) - crisp data
                or
            X (n_objects, n_objects) - pairwise distances if precomputed = True
            mus(n_objects)            - values of membership function
        """
        if precomputed:
            data_dist = X
        else:
            if not fuzzified:
                X_fuzzy = Fuzzifier(self.l, self.r, self.dc).fuzzify(X, mus)
            else:
                X_fuzzy = X
            checkpoint_time = time.time()
            # print('Calculating distances')
            data_dist = squareform(pdist3d(X_fuzzy, fuzzy_dist))  # matrix of distances
            # print('Distances calculated, %f' % (time.time() - checkpoint_time))
        #         self.dist_ = np.array(dist)
        sq_unq, unq_idx, unq_inv = np.unique(data_dist, axis=0, return_index=True, return_inverse=True)
        dist = sq_unq[:, unq_idx]
        del sq_unq
        # add one because you are your neighb.
        from sklearn.neighbors import NearestNeighbors
        checkpoint_time = time.time()
        # print('Finding neighbors', flush=True)
        nn = NearestNeighbors(n_neighbors=self.wishart_neighbors, metric='precomputed').fit(dist)
        distances, neighbors = nn.kneighbors(dist, return_distance=True)
        # print('Neighbors found, time %f' % (time.time() - checkpoint_time), flush=True)
        neighbors = neighbors[:, 1:]

        distances = distances[:, -1]
        indexes = np.argsort(distances)

        size, dim = X.shape[0], self.dim  # ???

        self.object_labels = np.zeros(size, dtype=int) - 1

        # ADDED FOR SCORES
        self.dk_ = distances

        # index in tuple
        # min_dist, max_dist, flag_to_significant
        self.clusters = np.array([(1., 1., 0)])
        self.clusters_to_objects = defaultdict(list)
        # print('Start clustering')

        for index in tqdm(indexes) if verbose else indexes:
            neighbors_clusters = \
                np.concatenate([self.object_labels[neighbors[index]], self.object_labels[neighbors[index]]])
            unique_clusters = np.unique(neighbors_clusters).astype(int)
            unique_clusters = unique_clusters[unique_clusters != -1]

            if len(unique_clusters) == 0:
                self._create_new_cluster(index, distances[index])
            else:
                max_cluster = unique_clusters[-1]
                min_cluster = unique_clusters[0]
                if max_cluster == min_cluster:
                    if self.clusters[max_cluster][-1] < 0.5:
                        self._add_elem_to_exist_cluster(index, distances[index], max_cluster)
                    else:
                        self._add_elem_to_noise(index)
                else:
                    my_clusters = self.clusters[unique_clusters]
                    flags = my_clusters[:, -1]
                    if np.min(flags) > 0.5:
                        self._add_elem_to_noise(index)
                    else:
                        significan = np.power(my_clusters[:, 0], -dim) - np.power(my_clusters[:, 1], -dim)
                        significan *= self.wishart_neighbors
                        significan /= size
                        significan /= np.power(np.pi, dim / 2)
                        significan *= gamma(dim / 2 + 1)
                        significan_index = significan >= self.significance_level

                        significan_clusters = unique_clusters[significan_index]
                        not_significan_clusters = unique_clusters[~significan_index]
                        significan_clusters_count = len(significan_clusters)
                        if significan_clusters_count > 1 or min_cluster == 0:
                            self._add_elem_to_noise(index)
                            self.clusters[significan_clusters, -1] = 1
                            for not_sig_cluster in not_significan_clusters:
                                if not_sig_cluster == 0:
                                    continue

                                for bad_index in self.clusters_to_objects[not_sig_cluster]:
                                    self._add_elem_to_noise(bad_index)
                                #                                 self.clusters_to_objects[not_sig_cluster].clear()
                                self.clusters_to_objects.pop(not_sig_cluster)
                        else:
                            for cur_cluster in unique_clusters:
                                if cur_cluster == min_cluster:
                                    continue

                                for bad_index in self.clusters_to_objects[cur_cluster]:
                                    self._add_elem_to_exist_cluster(bad_index, distances[bad_index], min_cluster)
                                #                                 self.clusters_to_objects[cur_cluster].clear()
                                self.clusters_to_objects.pop(cur_cluster)
                            self._add_elem_to_exist_cluster(index, distances[index], min_cluster)

        self.clean_data()    # CHANGED: SAVE LABELS IN SELF
        self.unq_inv = unq_inv
        self._relabel_data()
        return self

    def clean_data(self):
        unique = np.unique(self.object_labels)
        index = np.argsort(unique)
        if unique[0] != 0:
            index += 1
        true_cluster = {unq: index for unq, index in zip(unique, index)}
        labels_cleaned = np.zeros(len(self.object_labels), dtype = int)
        clusters_to_objects_cleaned = {}
        for index, unq in enumerate(self.object_labels):
            labels_cleaned[index] = true_cluster[unq]
            if unq in self.clusters_to_objects:
                clusters_to_objects_cleaned[true_cluster[unq]] = self.clusters_to_objects[unq]
                
        self.clusters_to_objects = clusters_to_objects_cleaned
        self.object_labels = labels_cleaned

    def _add_elem_to_noise(self, index):
        self.object_labels[index] = 0
        self.clusters_to_objects[0].append(index)

    def _create_new_cluster(self, index, dist):
        self.object_labels[index] = len(self.clusters)
        self.clusters_to_objects[len(self.clusters)].append(index)
        self.clusters = np.append(self.clusters, [(dist, dist, 0)], axis=0)

    def _add_elem_to_exist_cluster(self, index, dist, cluster_label):
        self.object_labels[index] = cluster_label
        self.clusters_to_objects[cluster_label].append(index)
        self.clusters[cluster_label][0] = min(self.clusters[cluster_label][0], dist)
        self.clusters[cluster_label][1] = max(self.clusters[cluster_label][1], dist)
        
    def _relabel_data(self):
        self.object_labels = self.object_labels[self.unq_inv]
        
        for cluster in self.clusters_to_objects:
            self.clusters_to_objects[cluster] = np.concatenate([
                np.where(self.unq_inv == idx)[0] for idx in self.clusters_to_objects[cluster]
            ])
