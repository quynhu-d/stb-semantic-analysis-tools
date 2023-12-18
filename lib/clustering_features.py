import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

def get_emb_n(data, n):
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
    d = X_dist[get_dist_idx(cluster_idx, n_max)]
    if len(d) == 0:
        return .0, .0, .0
    return d.mean(), d.min(), d.max()
    
def dist_chars(X_dist, labels, w_noise=False):
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
    return np.array([np.mean(means), np.mean(mins), np.mean(maxs)])

def get_kmeans_features(trajectory, n, k):
    X_n = get_emb_n(trajectory, n)
    X_dist = pdist(X_n)
    try:
        km = KMeans(n_clusters=k, random_state=123)
        km.fit(X_n)
    except:
        print(len(trajectory))
        print(len(X_n))
        raise
    return dist_chars(X_dist, km.labels_)