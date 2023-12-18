import numpy as np
from tqdm import tqdm

import gtda.homology as homology
from gtda.diagrams import Amplitude, PersistenceEntropy, NumberOfPoints
from sklearn.pipeline import Pipeline, make_union
from gtda.time_series import TakensEmbedding
from sklearn.preprocessing import FunctionTransformer

class TDAPipeline:
    def __init__(self, persistence_type='VietorisRipsPersistence', homology_dim=1, takens_params=None):
        transposer = FunctionTransformer(
            func=np.transpose, kw_args={'axes': (0,2,1)}, 
            inverse_func=np.transpose, inv_kw_args={'axes': (0,2,1)}
        )  # transpose data for tda
        if takens_params is None:
            takens_params = {
                "time_delay":1, "dimension":3, "stride":1, "flatten":True
            }
        embedder = TakensEmbedding(**takens_params)
        
        persistence = getattr(homology, persistence_type)(
            homology_dimensions=list(range(homology_dim + 1))
        )
        self.homology_dim = homology_dim        
        
        metrics = [
            {"metric": metric}
            for metric in ["bottleneck", "wasserstein", "landscape", "persistence_image"]
        ]
        feature_union = make_union(
            PersistenceEntropy(normalize=True),
            NumberOfPoints(n_jobs=-1),
            *[Amplitude(**metric, n_jobs=-1) for metric in metrics]
        )

        self.pipeline = Pipeline([
            ('transpose', transposer),
            ('takens_emb', embedder),
            ('tda_persistence', persistence),
            ('tda_features', feature_union)
        ])