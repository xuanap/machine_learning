"""Normalized spectral clustering for a preprocessed feature matrix."""

import warnings

import numpy as np
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_array


def spectral_clustering(
    X, *, n_clusters=4, affinity="nearest_neighbors", n_neighbors=20,
    gamma=0.1, random_state=42,
):
    """Return one integer label per row of X, in input order.

    The graph uses averaged directed kNN edges or RBF weights
    exp(-gamma * squared_distance). Self-loops are excluded. The
    bottom eigenvectors of the symmetric normalized Laplacian are
    row-normalized before K-means.
    """
    X = check_array(X, dtype=np.float64)
    n_samples = len(X)
    if not isinstance(n_clusters, (int, np.integer)) or not 2 <= n_clusters < n_samples:
        raise ValueError("n_clusters must be an integer in [2, n_samples).")

    if affinity == "nearest_neighbors":#k邻接
        if not isinstance(n_neighbors, (int, np.integer)) or not 1 <= n_neighbors < n_samples:
            raise ValueError("n_neighbors must be an integer in [1, n_samples).")
        directed = kneighbors_graph(X, n_neighbors, include_self=False) #连接1未连接0
        weights = (directed + directed.T) * 0.5
    elif affinity == "rbf":#全连接
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be finite and positive.")
        weights = rbf_kernel(X, gamma=gamma)
        np.fill_diagonal(weights, 0.0)
    else:
        raise ValueError("affinity must be 'nearest_neighbors' or 'rbf'.")

    components = connected_components(weights, directed=False, return_labels=False)#相似度图的连通分量
    if components > n_clusters:
        raise ValueError(
            f"The graph has {components} connected components for {n_clusters} clusters. "
            "Increase n_neighbors or decrease gamma."
        )
    if components > 1:
        warnings.warn(f"The affinity graph has {components} connected components.", stacklevel=2)

    # I - L_sym has the same eigenvectors as L_sym in reversed eigenvalue order.
    # Negating L_sym gives the same ordering without allocating an identity matrix.
    graph_laplacian = laplacian(weights, normed=True) #归一化得到拉普拉斯矩阵
    initial = np.random.default_rng(random_state).standard_normal(n_samples)
    _, eigenvectors = eigsh(-graph_laplacian, k=n_clusters, which="LA", v0=initial) #取最小的K个特征值对应的特征向量
    embedding = normalize(eigenvectors)#归一化
    return KMeans(
        n_clusters=n_clusters, n_init=20, random_state=random_state
    ).fit_predict(embedding) #重新运行kmeans的到客户标签
