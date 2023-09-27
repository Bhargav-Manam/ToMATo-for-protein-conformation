from sklearn.manifold import MDS
import numpy as np


def get_mds_embeddings(distances: np.ndarray) -> np.ndarray:
    """
    Compute the MDS embeddings in 2D for the given distance matrix.

    Args:
        distances (np.ndarray): A square matrix of pairwise distances.

    Returns:
        embeddings (np.ndarray): A 2D array of shape (n_samples, 2) representing the MDS embeddings.
    """
    mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress=False)
    embeddings = mds.fit_transform(distances)
    return embeddings


