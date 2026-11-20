"""K-means adapter: fit only, with no log transform, scaling, or PCA."""
import time
import warnings

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from .common import ClusteringResult, validate_matrix, validate_settings

DEFAULTS = {"n_clusters": 4, "init": "k-means++", "n_init": 10,
            "max_iter": 300, "tol": 1e-4, "algorithm": "lloyd"}


def fit_predict(X, params=None, random_state=42, threads=1):
    values = validate_matrix(X)
    settings = {**DEFAULTS, **(params or {})}
    validate_settings(settings, random_state, threads, "n_clusters", len(values))
    if not isinstance(settings["n_init"], int) or isinstance(settings["n_init"], bool) or settings["n_init"] < 1:
        raise ValueError("Use an explicit positive integer n_init for comparable experiments")
    model = KMeans(**settings, random_state=random_state)
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=threads):
        warnings.simplefilter("always")
        start = time.perf_counter()
        model.fit(values)
        seconds = time.perf_counter() - start
    distinct = len(np.unique(model.labels_))
    status = ("collapsed_clusters" if distinct < settings["n_clusters"] else
              "iteration_limit_reached" if model.n_iter_ >= settings["max_iter"] else "ok")
    # sklearn KMeans has no converged_ flag. Reaching max_iter is ambiguous;
    # preserve the fact instead of inventing an exact convergence judgement.
    return ClusteringResult(
        algorithm="kmeans", labels=model.labels_.copy(), centers=model.cluster_centers_.copy(),
        fit_seconds=seconds, params=model.get_params(), seed=random_state,
        status=status, n_iter=int(model.n_iter_), converged=None,
        warnings=[str(item.message) for item in captured],
        diagnostics={"inertia": float(model.inertia_), "distinct_clusters": distinct,
                     "iteration_limit_reached": bool(model.n_iter_ >= settings["max_iter"]),
                     "threads": threads},
    )
