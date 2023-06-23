import sys
import numpy as np
import pandas as pd
import glob
from tqdm.auto import tqdm
from scipy.spatial.distance import pdist
from fcmeans import FCM


def update_dict(d, new_d):
    for k, v in new_d.items():
        if k in d:
            d[k].append(v)


def clean_dict(d):
    for k in d:
        d[k] = []


def text2emb(text, wdict, m):
    embs = [wdict[w][-m:] for w in text.split() if w in wdict]
    if len(embs) == 0:
        return None
    return np.vstack(embs)


def get_emb_n(data, n):
    l = len(data)
    return np.unique(np.concatenate([data[i:l - n + i] for i in range(n)], axis=1), axis=0)


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
    return {
        'mean_icd': np.mean(means),
        'min_icd': np.mean(mins),
        'max_icd': np.mean(maxs),
    }


def main():
    m, n, k, emb_type, dict_path, file_path, table_path = sys.argv[1:]
    m = int(m)
    n = int(n)
    k = int(k)
    if emb_type.lower() == 'svd':
        wdict = np.load(dict_path, allow_pickle=True).item()
    else:
        wdict = np.load(dict_path % m, allow_pickle=True).item()
    metrics = ['mean_icd', 'min_icd', 'max_icd']
    fcm_table = {
        k: [] for k in [
                           'train/test', 'name', 'emb_type', 'text_type',
                           'm', 'n', 'k'
                       ] + metrics
    }
    n_buffer = 10
    files = glob.glob(file_path + '/lit/*')
    if 'train' in file_path.lower():
        files.extend(glob.glob(file_path + '/gpt2/*'))
        files.extend(glob.glob(file_path + '/balaboba/*'))
    else:  # test
        files.extend(glob.glob(file_path + '/mGPT/*'))
        files.extend(glob.glob(file_path + '/lstm/*'))

    for i, filename in tqdm(enumerate(files), total=len(files)):
        set_type, _, text_type, name = filename.split('_')  # Train_english_newlit_15.txt
        set_type = set_type[set_type.rfind('/') + 1:]  # train/test
        name = name[:-4]  # {num}.txt -> {num}
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        emb = text2emb(text, wdict, m)
        if emb is None:
            print(filename, '0', flush=True)
            continue
        try:
            X_n = get_emb_n(emb, n)[:2000]
        except:
            print(filename, '1', flush=True)
            continue
        try:
            X_dist = pdist(X_n)
            fcm = FCM(n_clusters=k)
            fcm.fit(X_n)
        except:
            print(filename, '2', flush=True)
            continue

        fcm_labels = fcm.predict(X_n)
        fcm_chars = dist_chars(X_dist, fcm_labels, w_noise=False)
        fcm_chars['k'] = k
        fcm_chars['m'] = m
        fcm_chars['n'] = n
        fcm_chars['name'] = name
        fcm_chars['text_type'] = text_type
        fcm_chars['emb_type'] = emb_type
        fcm_chars['train/test'] = set_type.lower()

        update_dict(new_d=fcm_chars, d=fcm_table)
        if ((i != 0) and (i % n_buffer == 0)) or (i == len(files) - 1):
            pd.DataFrame(fcm_table).to_csv(table_path, mode='a', index=False, header=False)
            clean_dict(fcm_table)


if __name__ == '__main__':
    main()
