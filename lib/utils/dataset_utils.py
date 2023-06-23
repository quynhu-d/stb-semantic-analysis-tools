import numpy as np
from tqdm.auto import tqdm, trange
from trajectories import get_corpus_trajectories


def get_dataset(text_data_path, lang, language, m, emb_type, wdict_path):
    svd_descending = (lang == 'VN')
    print("SVD descending:", svd_descending)
    lit_ts = get_corpus_trajectories(
        corpus_path=text_data_path+"Train/lit/*",
        wdict_path=wdict_path, 
        corpus_saved_as_file=False, wdim=m, svd_descending=svd_descending
    )

    gpt2_ts = get_corpus_trajectories(
        corpus_path=text_data_path+"Train/gpt2/*",
        wdict_path=wdict_path, 
        corpus_saved_as_file=False, wdim=m, svd_descending=svd_descending
    )

    balaboba_ts = get_corpus_trajectories(
        corpus_path=text_data_path+"Train/balaboba/*",
        wdict_path=wdict_path, 
        corpus_saved_as_file=False, wdim=m, svd_descending=svd_descending
    )

    lit_test_ts = get_corpus_trajectories(
        corpus_path=text_data_path+"Test/lit/*",
        wdict_path=wdict_path, 
        corpus_saved_as_file=False, wdim=m, svd_descending=svd_descending
    )

    mgpt_test_ts = get_corpus_trajectories(
        corpus_path=text_data_path+"Test/mGPT/*",
        wdict_path=wdict_path, 
        corpus_saved_as_file=False, wdim=m, svd_descending=svd_descending
    )

    lstm_test_ts = get_corpus_trajectories(
        corpus_path=text_data_path+"Test/lstm/*",
        wdict_path=wdict_path, 
        corpus_saved_as_file=False, wdim=m, svd_descending=svd_descending
    )
    
    return {
        "lit_1": lit_ts,
        "lit_2": lit_test_ts,
        "balaboba": balaboba_ts,
        "mGPT": mgpt_test_ts,
        "gpt2": gpt2_ts,
        "lstm": lstm_test_ts
    }

from sklearn.model_selection import train_test_split
def split_data(dataset, train_set=["lit_1", "balaboba", "gpt2"], test_set=["lit_2", "mGPT", "lstm"], split_all=False):
    if split_all:
        target = np.concatenate([
            np.full(len(dataset[d]), d[:-2] if 'lit' in d else d) for d in dataset 
        ])
        
        min_len = np.min([dataset[d].shape[1] for d in dataset])
        total_ts = np.vstack([dataset[d][:,:min_len] for d in dataset])
        train_ts, test_ts, y_train, y_test = train_test_split(
            total_ts, target, test_size=1200/5200, random_state=42, shuffle=True, stratify=target
        )
    else:
        min_len = np.min([dataset[d].shape[1] for d in train_set])
        train_ts = np.vstack([dataset[d][:,:min_len] for d in train_set])
        
        min_len = np.min([dataset[d].shape[1] for d in test_set])
        test_ts = np.vstack([dataset[d][:,:min_len] for d in test_set])

        y_train = np.concatenate([
            np.full(len(dataset[d]), d[:-2] if 'lit' in d else d) for d in train_set 
        ])
        y_test = np.concatenate([
            np.full(len(dataset[d]), d[:-2] if 'lit' in d else d) for d in test_set 
        ])
    return (train_ts, y_train), (test_ts, y_test)

def test_dataset_loading():
    dataset_params = {
        "text_data_path":"../DATASET/English/",
        "lang":"EN",
        "language":"english",
        "m":1,
        "emb_type":"SVD",
        "wdict_path":"/home/kdang/EN/english_data/english_newlit_SVD_dict.npy"
    }
    get_dataset(**dataset_params)
    

if __name__ == '__main__':
    test_dataset_loading()