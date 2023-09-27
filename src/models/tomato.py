from gudhi.clustering.tomato import Tomato
import numpy as np


def fit_tomato_model(embeddings: np.ndarray, k: int = 10, merge_threshold: float=None, density_type="logDTM", n_clusters=None):
    """
    Fits a Tomato clustering model to the given embeddings.

    Args:
    - embeddings: np.ndarray, shape=(n_samples, n_features)
        The input data to cluster.
    - k: int, default=10
        The number of clusters to construct.
    - merge_threshold: float or None, default=None
        Minimum prominence of a cluster, so it doesn't get merged.
    - density_type: {'manual', 'DTM', 'logDTM', 'KDE', 'logKDE'}, default='logDTM'
        The type of density measure to use.
        When you have many points, 'KDE' and 'logKDE' tend to be slower. Default is 'logDTM'.
    - n_clusters: int or None, default=None
        Number of clusters to perform ToMaTo for. Defaults to None, i.e. no merging occurs, and we get
        the maximal number of clusters.

    Returns:
    - t: Tomato object
        The fitted Tomato clustering model.
    """
    t = Tomato(n_jobs=-1, k=k, merge_threshold=merge_threshold, density_type=density_type, n_clusters=n_clusters)
    t.fit(embeddings)
    return t
