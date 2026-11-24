"""B's hierarchical adapter. X must already be preprocessed by C."""
import time

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage as make_linkage
from threadpoolctl import threadpool_limits

from .common import ClusteringResult, validate_matrix, validate_settings


def fit_predict(X, params=None, random_state=42, threads=1):
    """Return the shared result contract; preserve input coordinates and row order.

    The full linkage tree is available in diagnostics['linkage_matrix'] for
    dendrograms. fit_seconds includes building the tree and cutting it at K.
    """
    values = validate_matrix(X)
    settings = {"n_clusters": 4, "linkage": "ward", **(params or {})}
    if set(settings) - {"n_clusters", "linkage"}:
        raise ValueError("Supported parameters: n_clusters, linkage")
    validate_settings(settings, random_state, threads, "n_clusters", len(values))
    if settings["linkage"] not in {"ward", "complete", "average"}:
        raise ValueError("linkage must be ward, complete or average")
    with threadpool_limits(limits=threads):
        start = time.perf_counter()
        tree = make_linkage(values, method=settings["linkage"], metric="euclidean")
        labels = cut_tree(tree, n_clusters=[settings["n_clusters"]]).ravel().astype(int)
        seconds = time.perf_counter() - start
    centers = np.array([values[labels == label].mean(axis=0) for label in np.unique(labels)])
    return ClusteringResult(
        algorithm="hierarchical", labels=labels, centers=centers,
        fit_seconds=seconds, params=settings, seed=random_state, status="ok",
        n_iter=0, converged=None,
        diagnostics={"linkage_matrix": tree, "threads": threads,
                     "metric": "euclidean", "deterministic": True},
    )


def hierarchical_clustering(X, n_clusters=4, linkage="ward"):
    """Labels-only interface for C, matching spectral_clustering's convention."""
    return fit_predict(X, {"n_clusters": n_clusters, "linkage": linkage}).labels
