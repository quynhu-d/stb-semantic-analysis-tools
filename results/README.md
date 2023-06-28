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

## Final classification

Train dataset:
- 2000 literary texts
- 1000 balaboba-generated texts
- 1000 gpt2-generated texts

Test dataset:
- 600 literary texts
- 300 lstm-generated texts
- 300 mgpt-generated texts

[Decision tree classifier](res_clf_decision_tree.csv):
|method |rus  |en   |ger  |vn   |fr   |
|-------|-----|-----|-----|-----|-----|
|ec     |0.769|0.824|0.971|0.948|0.642|
|wishart|0.562|0.705|0.710|0.646|0.635|
|fuzzy  |0.697|0.853|0.862|0.878|0.915|
|kmeans |0.973|0.860|0.633|0.666|0.691|
|fcmeans|0.930|0.782|0.675|0.730|0.635|
|all    |0.980|0.878|0.897|0.720|0.858|

[Random forest classifier](res_clf_random_forest.csv)
|method |rus  |en   |ger  |vn   |fr   |
|-------|-----|-----|-----|-----|-----|
|ec     |0.780|0.828|0.976|0.970|0.865|
|wishart|0.550|0.733|0.715|0.665|0.605|
|fuzzy  |0.695|0.854|0.891|0.813|0.926|
|kmeans |0.977|0.868|0.613|0.670|0.510|
|fcmeans|0.947|0.777|0.602|0.721|0.671|
|all    |0.992|0.908|0.912|0.717|0.857|

