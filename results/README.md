# Results directory

For each method you can find:
```
.
└── {method}/
    ├── best_classification_params/ # best parameters for each classification model
    │   ├── stb_allvslit_{method}_clf.csv # binary classification (all bots vs literature), is used in the final experiment
    │   ├── stb_all_{method}_clf.csv # multiclass classification
    │   ├── stb_lstm_{method}_clf.csv # one type of bot vs literature
    │   ├── ...
    │   └── stb_mGPT_{method}_clf.csv
    ├── {method}_feature_values/ # features derived from each method for the final experiment
    │   ├── english_DATASET_{method}.csv
    │   ├── ...
    │   └── vietnamese_DATASET_{method}.csv
    └── {method}_clf/ # classification accuracy values for different parameter sets
        ├── english/clf_tables
        ├── ...
        └── vietnamese/clf_tables
```
