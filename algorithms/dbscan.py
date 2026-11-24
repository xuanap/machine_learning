"""B's DBSCAN adapter; noise remains -1 and X is never rescaled."""
import time

import numpy as np
from sklearn.cluster import DBSCAN
from threadpoolctl import threadpool_limits

from .common import ClusteringResult, validate_matrix, validate_settings


def fit_predict(X, params=None, random_state=42, threads=1):
    values = validate_matrix(X)
    settings = {"eps": 0.65, "min_samples": 10, **(params or {})}
    if set(settings) - {"eps", "min_samples"}:
        raise ValueError("Supported parameters: eps, min_samples")
    validate_settings({"n_clusters": 1}, random_state, threads, "n_clusters", len(values))
    eps, minimum = settings["eps"], settings["min_samples"]
    if isinstance(eps, bool) or not np.isscalar(eps) or not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        raise ValueError("min_samples must be a positive integer")
    model = DBSCAN(**settings, metric="euclidean", n_jobs=threads)
    with threadpool_limits(limits=threads):
        start = time.perf_counter()
        model.fit(values)
        seconds = time.perf_counter() - start
    labels = model.labels_.astype(int, copy=True)
    groups = np.unique(labels[labels != -1])
    centers = (np.array([values[labels == group].mean(axis=0) for group in groups])
               if len(groups) else np.empty((0, values.shape[1])))
    return ClusteringResult(
        algorithm="dbscan", labels=labels, centers=centers, fit_seconds=seconds,
        params=settings, seed=random_state, status="ok", n_iter=0, converged=None,
        diagnostics={"threads": threads, "metric": "euclidean", "deterministic": True,
                     "effective_clusters": len(groups),
                     "core_samples": len(model.core_sample_indices_)},
    )


def dbscan_clustering(X, eps=0.65, min_samples=10):
    """Labels-only interface; includes -1 entries in original input row order."""
    return fit_predict(X, {"eps": eps, "min_samples": min_samples}).labels
