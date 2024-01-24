import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from clustering.WishartParallelKD import Wishart
from clustering.WishartFUZZY import Wishart_fuzzy
from fcmeans import FCM
from ordec.ordec import entropy_complexity


def get_entropy_complexity_features(trajectory, args):
    """
    Pipeline for feature retrieval based on entropy-complexity.

    Params:
        trajectory (ndarray): consecutive word embeddings
        args
    
    Returns:
        entropy-complexity features to be passed to classification
    """
    X_n = get_emb_n(trajectory, args.n)
    try:
        ent, comp = entropy_complexity(X_n, m=args.wdim, n=args.n)
    except:
        print(len(trajectory))
        print(len(X_n))
        raise
    return np.array([ent, comp])


def get_cluster_model(args, seed=123):
    """
    Wrapper function to get clustering model with set arguments.
    """
    if args.method == 'kmeans':
        return KMeans(args.k, random_state=seed)
    if args.method == 'wishart':
        return Wishart(args.k, .1)
    if args.method == 'fcmeans':
        return FCM(n_clusters=args.k)
    if args.method == 'fwishart':
        return Wishart_fuzzy(args.k, .1, dim=args.dim * args.n)

def get_cluster_labels(model, X_n, args):
    """
    Wrapper function to get clustering labels.
    """
    if args.method == 'kmeans':
        return model.labels_
    if 'wishart' in args.method:
        return model.object_labels
    if args.method == 'fcmeans':
        return model.predict(X_n)

def get_emb_n(data, n):
    """
    Get concatenated embeddings for n-grams.

    Params:
        data (ndarray): array of shape (text length x wdim)
        n (int): number of words in n-gram
    Returns:
        ndarray of shape ((text_length - n + 1) x (wdim * n)), consecutive ngram embeddings
    """
    l = len(data)
    return np.concatenate([data[i:l - n + i+1] for i in range(n)], axis=1)

def get_dist_idx(idx, n_max=1000):
    idx = sorted(idx)
    d_idx = []
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            i = idx[ii]
            j = idx[jj]
            d_idx.append(n_max * i + j - ((i + 2) * (i + 1)) // 2)
    return d_idx
    
def cluster_distances(X_dist, cluster_idx, n_max):
    """
    Get intracluster distances.

    Params:
        X_dist (ndarray): array of shape (text length x text length), pairwise distances
        cluster_idx (int): index of cluster label
        n_max (int): max length of pairwise array
    
    Returns:
        mean, min, max intracluster distances
    """
    d = X_dist[get_dist_idx(cluster_idx, n_max)]
    if len(d) == 0:
        return .0, .0, .0
    return d.mean(), d.min(), d.max()
    
def dist_chars(X_dist, labels, w_noise=False):
    """
    Get intracluster characteristics.

    Params:
        X_dist (ndarray): array of shape (text length x text length), pairwise distances
        labels (ndarray): array of shape (text length), clustering labels
        w_noise (bool): if True, noise cluster (with label 0) is not included
                        additional features: number of clusters, noise ratio
    Returns:
        mean intracluster distances by all clusters
    """
    means, mins, maxs = [], [], []
    n_max = int((1 + np.sqrt(1 + X_dist.shape[0] * 8)) // 2)
    for c in set(labels):
        if (c == 0) and w_noise:
            continue
        cluster_idx = np.where(labels == c)[0]
        mean, min1, max1 = cluster_distances(X_dist, cluster_idx, n_max)
        means.append(mean)
        mins.append(min1)
        maxs.append(max1)
    result = [np.mean(means), np.mean(mins), np.mean(maxs)]
    if w_noise:
        result.append(len(np.unique(labels)) - 1)
        result.append((labels == 0).mean())
    return np.array(result)

def get_clustering_features(trajectory, args):
    """
    Pipeline for clustering feature retrieval.

    Params:
        trajectory (ndarray): consecutive word embeddings
        args
    
    Returns:
        clustering features to be passed to classification
    """
    X_n = get_emb_n(trajectory, args.n)
    X_dist = pdist(X_n)
    cluster_model = get_cluster_model(args)
    try:
        cluster_model.fit(X_n)
        labels = get_cluster_labels(cluster_model, X_n, args)
    except:
        print(len(trajectory))
        print(len(X_n))
        raise
    return dist_chars(X_dist, labels, w_noise="wishart" in args.method)
