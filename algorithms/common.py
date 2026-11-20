"""Result contract shared by K-means and GMM."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClusteringResult:
    algorithm: str
    labels: np.ndarray
    centers: np.ndarray
    fit_seconds: float
    params: dict
    seed: int
    status: str
    n_iter: int
    converged: bool | None
    warnings: list[str] = field(default_factory=list)
    probabilities: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)


def validate_matrix(X):
    if hasattr(X, "columns") and "CustomerID" in X.columns:
        raise ValueError("Remove CustomerID before calling the algorithm")
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("X must be a 2D matrix with at least two samples and one feature")
    if not np.isfinite(values).all():
        raise ValueError("X contains missing or infinite values")
    return values


def validate_settings(params, seed, threads, cluster_key, n_samples):
    if "random_state" in params:
        raise ValueError("Supply the random seed with random_state, not inside params")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("random_state must be an integer in [0, 2**32)")
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")
    count = params[cluster_key]
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= n_samples:
        raise ValueError(f"{cluster_key} must be between 1 and the number of samples")
