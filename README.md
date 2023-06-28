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
├── lib          # includes full pipelines/
│   ├── ec
│   ├── kmeans
│   ├── fcmeans
│   ├── wishart
│   ├── fuzzy
│   └── tda
├── examples     # includes examples for implemented methods
└── results      # includes resulting tables
```
