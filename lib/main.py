import argparse
import json
import pickle
from tqdm.auto import tqdm

from preprocessing.split import split_to_paragraphs
from preprocessing.text_preprocessing import prepare_russian_text, prepare_english_text
from trajectories import trajectory_from_text
import numpy as np
from features import get_clustering_features, get_entropy_complexity_features


def main():
    args = get_arguments()
    with open(args.input_path, 'r', encoding='utf-8') as fin:
        text = fin.read()
    with open(args.clf_path, 'rb') as f:
        classifier = pickle.load(f)
    print("Classification model loaded:", classifier, flush=True)

    if args.split_text:
        paragraphs = split_to_paragraphs(text, lang=args.lang)
        print(f"Texts split into {len(paragraphs)} paragraphs.")
    else:
        paragraphs = [text]
    if args.lang == 'english':
        preprocessing_f = prepare_english_text
    elif args.lang == 'russian':
        preprocessing_f = prepare_russian_text
    preprocessed_paragraphs = [preprocessing_f(paragraph) for paragraph in tqdm(paragraphs, desc='Preprocessing paragraphs...')]

    wdict = np.load(args.wdict_path, allow_pickle=True).item()

    trajectories = [
        trajectory_from_text(paragraph.split(), wdict, args.wdim, text_length_threshold=None) 
        for paragraph in tqdm(preprocessed_paragraphs, desc='Retrieving trajectories...')
    ]
    trajectory_lengths = np.array(list(map(len, trajectories)))
    paragraph_idx = np.where(trajectory_lengths >= args.k + args.n - 1)[0]
    features_f = get_entropy_complexity_features if args.method == 'ec' else get_clustering_features
    features = np.array([
        features_f(traj, args) 
        for traj in tqdm(np.array(trajectories)[paragraph_idx], desc='Getting features...')
    ])
    predictions = classifier.predict(features)
    if len(predictions) == 1:
        results = {'text': paragraphs[0], 'is_bot': bool(~predictions[0])}
    else:
        overall_prediction = predictions.mean()
        results = {'is_bot_prob': 1 - overall_prediction, 'per_paragraph':[]}
        for pred, paragraph in zip(predictions, np.array(paragraphs)[paragraph_idx]):
            results['per_paragraph'].append({
                'paragraph': paragraph,
                'is_bot': bool(~pred)
            })
    with open(args.save_prediction_path, 'w') as f:
        json.dump(results, f)
    print(f'Results saved at {args.save_prediction_path}.')
    

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="samples/sample.txt")
    parser.add_argument('--lang', type=str, default="english", choices=['english', 'russian'])
    parser.add_argument('--method', type=str, default="kmeans", choices=['kmeans', 'wishart', 'fcmeans', 'ec'])
    parser.add_argument('--wdict_path', type=str, default="data/english_lit_SVD_dict.npy")
    parser.add_argument('-wdim', type=int, default=8)
    parser.add_argument('-n', type=int, default=2)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('--clf_path', type=str, default="data/en_km_rf.pkl")
    parser.add_argument('--split_text', action="store_true")
    
    parser.add_argument('--save_prediction_path', type=str, default='sample_predictions.json')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()