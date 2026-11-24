"""Shared adapters returning one integer cluster label per input row."""

import numpy as np
from threadpoolctl import threadpool_limits

from algorithms import gmm, kmeans
from algorithms.common import validate_matrix
from algorithms.spectral import spectral_clustering


def _spectral(X, params, seed, threads):
    return spectral_clustering(X, **params, random_state=seed)


ADAPTERS = {
    "kmeans": lambda X, params, seed, threads: kmeans.fit_predict(X, params, seed, threads).labels,
    "gmm": lambda X, params, seed, threads: gmm.fit_predict(X, params, seed, threads).labels,
    "spectral": _spectral,
}


def cluster(X, algorithm, params, *, random_state=42, threads=1):
    """Run a registered adapter. The label -1 is reserved for noise."""
    values = validate_matrix(X)
    with threadpool_limits(limits=threads):
        labels = np.asarray(ADAPTERS[algorithm](values, dict(params), random_state, threads))
    if labels.shape != (len(values),) or labels.dtype.kind not in "iu" or (labels < -1).any():
        raise ValueError("Expected one integer label >= -1 per input row")
    return labels
