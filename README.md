# Spot the Bot: Semantic Analysis of Natural Language Paths

Used techniques:
- Clustering:

  algorithm type|crisp  |fuzzy        
  --------------|-------|-------------
  centroid-based|K-Means|C-Means      
  density-based |Wishart|Fuzzy Wishart
  
- Information theory
  
  Entropy-Complexity of ordinal patterns
- Topological data analysis

  Vietoris-Rips filtration + H0- and H1-diagram features

```
.
├── lib                      # includes full pipelines/
│   ├── trajectories         # text to semantic trajectory
│   ├── ordec                # implemented method for entropy-complexity calculations
│   ├── clustering           # clustering pipelines/
│   │   ├── pipelines
│   │   └── WishartFUZZY.py  # implemented Wishart algorithm for fuzzified data
│   └── tda                  # features using tda                   
├── examples                 # includes examples for implemented methods
└── results                  # includes resulting tables
```

## Full pipeline

Black-box solution:
TODO: add fuzzy wishart features, model retraining, hyperparameter selection.

```bash
python lib/main.py --input_path="examples/The Picture of Dorian Gray.txt" --save_prediction_path=sample_predictions.json
```

Pipeline parameters:

- `input_path`: path to text file
- `lang`: `english/russian`, language of the text
- `split_text`: if set, split text into paragraphs and predict per paragraph

- `wdict_path`: path to word dictionary with .npy extension
- `wdim`: word embedding dimension (8 by default)
- `n`: number of words in ngram (2 by default)
- `method`: clustering method (kmeans, fcmeans, wishart) or entropy-complexity method ('ec')
- `k`: number of clusters in kmeans/fcmeans; number of neighbours in Wishart
- `clf_path`: path to pretrained sklearn classifier model, .pkl extension
- `save_prediction_path`: path to save results as json file
