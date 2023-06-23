import sys
import numpy as np
import pandas as pd
import os
import glob
import errno
from tqdm.auto import tqdm, trange

from ordec import entropy_complexity


def update_dict(dict, new_row):
    for k, v in new_row.items():
        dict[k].append(v)
    return


def clean_dict(d):
    for k in d:
        d[k] = []


def text2emb(text, wdict, m, svd_inv_flag=True):
    if svd_inv_flag:
        embs = [wdict[w][-m:] for w in text.split() if w in wdict]
    else:
        embs = [wdict[w][:m] for w in text.split() if w in wdict]
    if len(embs) == 0:
        return None
    return np.vstack(embs)


def main():
    m, n, emb_type, dict_path, file_path, table_path = sys.argv[1:]
    m = int(m)
    n = int(n)
    svd_inv_flag = not ('vietnamese' in table_path.lower())
    print('SVD embeddings are inverted:', svd_inv_flag, flush=True)
    if emb_type.lower() == 'svd':
        wdict = np.load(dict_path, allow_pickle=True).item()
    else:
        wdict = np.load(dict_path % m, allow_pickle=True).item()
    n_buffer = 10
    table_dict = {
        'train/test': [], 'name': [], 'txt_type': [], 'emb_type': [],
        'n': [], 'm': [], 'entropy': [], 'complexity': []
    }
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
        emb = text2emb(text, wdict, m, svd_inv_flag)
        if emb is None:
            print(filename, flush=True)
            continue
        e, c = entropy_complexity(emb, m=m, n=n)
        update_dict(
            table_dict,
            {
                'train/test': set_type.lower(), 'name': name, 'txt_type': text_type,
                'emb_type': emb_type, 'n': n, 'm': m, 'entropy': e, 'complexity': c
            }
        )
        if (i != 0) and (i % n_buffer == 0):
            pd.DataFrame(table_dict).to_csv(table_path, mode='a', index=False, header=False)
            clean_dict(table_dict)
    pd.DataFrame(table_dict).to_csv(table_path, mode='a', index=False, header=False)
    clean_dict(table_dict)
    return


if __name__ == '__main__':
    main()
