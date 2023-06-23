import numpy as np
from tqdm.auto import tqdm, trange
import sys
import errno
import glob
import os
import time
from scipy.spatial.distance import pdist, squareform, euclidean
from itertools import product
from numpy.linalg import norm

import pandas as pd
from WishartFUZZY import Wishart_fuzzy


def text2emb(text, wdict, m):
    embs = [wdict[w][-m:] for w in text.split() if w in wdict]
    if len(embs) == 0:
        return None
    return np.vstack(embs)


def get_ngram_data(X, n):
    """
        m-dimensional embeddings to ngram data
    """
    data = []
    for i in range(n):
        data.append(X[i:len(X) - n + i + 1])
    return np.hstack(data)


def calculate_mu(X, m=8, n=2, normalize=True):
    if len(X.shape) != 2:
        try:
            X = X.reshape((-1, m))
        except:
            print('X could not be reshaped, m=%d' % m)
            raise
    assert ((len(X.shape) == 2) and (X.shape[-1] == m))

    columns = []
    for i, column in enumerate(X.T):
        unique_counts = np.unique(column, return_counts=True)
        d = {k: v for k, v in zip(unique_counts[0], unique_counts[1])}  # replace values with counts
        if normalize:
            arr = np.array([d[x] for x in column])
            columns.append(arr / arr.max())  # normalize by most frequent word
        else:
            columns.append(np.array([d[x] for x in column]))
        del d
    return np.column_stack(columns).reshape((-1, n, m))


def get_fuzzy_number(X, mu, l=1e-3, r=1e-3, dc=1e-2, method='left'):
    """
        Builds a fuzzy number based on data points and corresponding \mu values.

        X: data points
        mu: corresponding mu values
        l: length of left slope
        r: length of right slope
        dc: length between fuzzy number centers
        method: if 'left'/'right',
                fuzzy number is calculated with a presumption that mu fall on the left/right slope;
                if 'random', ...
    """
    if method == 'left':
        m1 = (1 - mu) * l + X
        m2 = m1 + dc
    if method == 'right':
        m2 = X - (1 - mu) * r
        m1 = m2 - dc
    return m1, m2, l, r


def overlap_mu(m1_1, m1_2, m2_1, m2_2, l=1e-3, r=1e-3):
    right_end = min(m1_2 + r, m2_2 + r)
    left_end = max(m1_1 - l, m2_1 - l)
    if right_end - left_end <= 0:
        return (0, 0, 0, 0, 0)

    if left_end == (m1_1 - l):
        if m1_1 < m2_2:
            new_m1 = m1_1
            new_m2 = m2_2
            new_max_mu = 1.0
        else:
            intersection = (m1_1 * r + m2_2 * l) / (l + r)
            new_m1 = intersection
            new_m2 = intersection
            new_max_mu = 1 - (m1_1 - intersection) / l
    else:
        if m2_1 < m1_2:
            new_m1 = m2_1
            new_m2 = m1_2
            new_max_mu = 1.0
        else:
            intersection = (m2_1 * r + m1_2 * l) / (l + r)
            new_m1 = intersection
            new_m2 = intersection
            new_max_mu = 1 - (m2_1 - intersection) / l
    new_l = new_m1 - left_end
    new_r = right_end - new_m2
    return new_m1, new_m2, new_l, new_r, new_max_mu


def get_overlap_mu(m1, m2, m, l, r):
    """
    Returns: fuzzy data of shape B x 5 x m
    [m1, m2, l, r, mu_max] for each component of each fuzzy number
    """
    fuzzy_data = []
    for ngram in zip(m1, m2):
        obj1 = ngram[0][0], ngram[1][0]
        obj2 = ngram[0][1], ngram[1][1]
        new_m1_row = []
        new_m2_row = []
        new_l_row = []
        new_r_row = []
        new_mu_max = []
        for i in range(m):
            m1_1, m2_1 = obj1[0][i], obj1[1][i]  # object 1
            m1_2, m2_2 = obj2[0][i], obj2[1][i]  # object 2
            new_m1, new_m2, new_l, new_r, new_max_mu = overlap_mu(m1_1, m1_2, m2_1, m2_2, l, r)
            new_m1_row.append(new_m1)
            new_m2_row.append(new_m2)
            new_mu_max.append(new_max_mu)
            new_l_row.append(new_l)
            new_r_row.append(new_r)
        fuzzy_data.append([new_m1_row, new_m2_row, new_l_row, new_r_row, new_mu_max])
    return np.array(fuzzy_data)


def fuzzy_distance(x_1, x_2, lambda_=.5, ro_=.5, L=None, R=None):  # default lambda = ro = .5 - trapezoidal mfunc
    m1_1, m2_1, l_1, r_1, max_mu1 = x_1
    m1_2, m2_2, l_2, r_2, max_mu2 = x_2
    if lambda_ is None:
        if L is not None:
            raise NameError('No inverse func')
            # L_inv = inversefunc(L)
            # lambda_ = quad(L_inv, 0, 1)[0]
        else:
            raise NameError('No information about L-side')
    if ro_ is None:
        if R is not None:
            raise NameError('No inverse func')
            # R_inv = inversefunc(R)
            # ro_ = quad(R_inv, 0, 1)[0]
        else:
            raise NameError('No information about R-side')

    return np.sqrt(
        norm(m1_1 - m1_2) ** 2 + norm(m2_1 - m2_2) ** 2 + \
        norm((m1_1 - lambda_ * l_1) - (m1_2 - lambda_ * l_2)) ** 2 + \
        norm((m2_1 - ro_ * r_1) - (m2_2 - ro_ * r_2)) ** 2
    )


def pdist3d(fuzzy_data, metric):
    k = 0
    m = len(fuzzy_data)
    dm = np.zeros((m * (m - 1)) // 2)
    for i in trange(m - 1, leave=False, desc='Calculating distances...'):
        for j in range(i + 1, m):
            dm[k] = metric(fuzzy_data[i], fuzzy_data[j])
            k += 1
    return dm


def update_dict(d, new_d):
    for k, v in new_d.items():
        d[k].append(v)


def clean_dict(d):
    for k in d:
        d[k] = []


def get_dist_idx(idx, n_max=1000):
    idx = sorted(idx)
    d_idx = []
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            i = idx[ii]
            j = idx[jj]
            d_idx.append(n_max * i + j - ((i + 2) * (i + 1)) // 2)
    return d_idx


def cluster_distances(X_dist, cluster_idx, n_max=1000):
    d = X_dist[get_dist_idx(cluster_idx, n_max)]
    if len(d) == 0:
        return .0, .0, .0
    return d.mean(), d.min(), d.max()


def fuzzy_chars(X_dist, wf):
    means, mins, maxs = [], [], []
    n_max = int((1 + np.sqrt(1 + X_dist.shape[0] * 8)) // 2)
    for c, cluster_idx in wf.clusters_to_objects.items():
        if (c != 0) and (len(cluster_idx) > 0):
            mean, min1, max1 = cluster_distances(X_dist, cluster_idx, n_max)
            means.append(mean)
            mins.append(min1)
            maxs.append(max1)
    if 0 in wf.clusters_to_objects:
        noise_ratio = len(wf.clusters_to_objects[0]) / len(wf.object_labels)
        n_clusters = len(wf.clusters_to_objects) - 1
    else:
        noise_ratio = 0.0
        n_clusters = len(wf.clusters_to_objects)
    return {
        'mean_icd': np.mean(means),
        'min_icd': np.mean(mins),
        'max_icd': np.mean(maxs),
        'noise_ratio': noise_ratio,
        'n_clusters': n_clusters
    }


def main():
    m, n, k, emb_type, dict_path, file_path, table_path = sys.argv[1:]
    m = int(m)
    n = int(n)
    k = int(k)
    l = 1e-4
    r = 1e-4
    if emb_type.lower() == 'svd':
        wdict = np.load(dict_path, allow_pickle=True).item()
    else:
        wdict = np.load(dict_path % m, allow_pickle=True).item()
    fc_table = {k: [] for k in [
        'train/test', 'name', 'emb_type', 'text_type',
        'm', 'n', 'k',
        'mean_icd', 'min_icd', 'max_icd',
        'noise_ratio', 'n_clusters'
    ]}
    n_buffer = 10
    files = sorted(glob.glob(file_path + '/lit/*'))
    if 'train' in file_path.lower():
        files.extend(sorted(glob.glob(file_path + '/gpt2/*')))
        files.extend(sorted(glob.glob(file_path + '/balaboba/*')))
    else:  # test
        files.extend(sorted(glob.glob(file_path + '/mGPT/*')))
        files.extend(sorted(glob.glob(file_path + '/lstm/*')))

    for i, filename in tqdm(enumerate(files), total=len(files)):
        set_type, _, text_type, name = filename.split('_')  # Train_english_newlit_15.txt
        set_type = set_type[set_type.rfind('/') + 1:]  # train/test
        name = name[:-4]  # {num}.txt -> {num}
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        if len(text) < k + 1:
            print(filename, '-1', flush=True)
            continue
        emb = text2emb(text, wdict, m)
        if emb is None:
            print(filename, '0', flush=True)
            continue
        X = get_ngram_data(emb, n)
        X = X.reshape(-1, n, m)
        if X.shape[0] < k + 1:
            print(filename, '1', flush=True)
            continue

        ngram_mu = calculate_mu(X, m, n)
        m1, m2, l, r = get_fuzzy_number(X, ngram_mu, l=l, r=r, dc=1e-2)
        if n == 2:
            fuzzy_data = get_overlap_mu(m1, m2, m, l, r)
        elif n == 1:
            fuzzy_data = np.hstack([
                m1.reshape(-1, m), m2.reshape(-1, m),
                np.full((len(m1), m), l), np.full((len(m1), m), r),
                np.ones((len(m1), m))
            ]).reshape(-1, 5, m)

        fuzzy_data = fuzzy_data[:2000]
        dist = pdist3d(fuzzy_data, fuzzy_distance)  # distances
        sq_data = squareform(dist)
        try:
            wf = Wishart_fuzzy(k, .1, dim=m * n)
            wf.fit(sq_data, precomputed=True, verbose=False)
            wf_chars = fuzzy_chars(dist, wf)

            wf_chars['k'] = k
            wf_chars['m'] = m
            wf_chars['n'] = n
            wf_chars['name'] = name
            wf_chars['text_type'] = text_type
            wf_chars['emb_type'] = emb_type
            wf_chars['train/test'] = set_type.lower()

            update_dict(new_d=wf_chars, d=fc_table)
            if ((i != 0) and (i % n_buffer == 0)) or (i == len(files) - 1):
                pd.DataFrame(fc_table).to_csv(table_path, mode='a', index=False, header=False)
                clean_dict(fc_table)
        except:
            print(filename, '2', flush=True)
            continue

if __name__ == "__main__":
    main()
