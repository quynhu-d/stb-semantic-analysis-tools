import numpy as np
from tqdm.auto import tqdm, trange

def corpus_length(corpus_path):
    corpus_l = 0
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for l in f:
            corpus_l += 1
    return corpus_l

def get_corpus_trajectories(
    corpus_path="../../STB/DATA/english_data/english_newlit_corpus.txt",
    wdict_path="../../STB/DATA/english_data/english_newlit_SVD_dict.npy",
    nrows=None, text_length_threshold=None, wdim=None, svd_descending=False,
    clip=True
):
    try:
        wdict = np.load(wdict_path, allow_pickle=True).item()
        if svd_descending:
            wdict = {w:e[::-1] for w, e in wdict.items()}
    except:
        print(f'No word embedding dictionary found at path {wdict_path}.')
    corpus_len = corpus_length(corpus_path)
    with open(corpus_path, 'r', encoding='utf-8') as f:
        time_series = []
        for line in tqdm(f, total=corpus_len, desc='Reading corpus...'):
            text = line.split()
            if (nrows is not None) and (len(time_series) == nrows):
                break
            if (text_length_threshold is not None) and (len(text) < text_length_threshold):
                continue
            ts = np.array([wdict[w][:wdim] for w in text if w in wdict])
            time_series.append(ts)
    if clip:
        min_len = min(list(map(len, time_series)))
        return np.array([ts[:min_len] for ts in time_series])
    return time_series
