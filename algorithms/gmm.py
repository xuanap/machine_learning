"""Gaussian mixture adapter with posterior memberships and convergence facts."""
import time
import warnings

import numpy as np
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from .common import ClusteringResult, validate_matrix, validate_settings

DEFAULTS = {"n_components": 4, "covariance_type": "full", "tol": 1e-3,
            "reg_covar": 1e-6, "max_iter": 300, "n_init": 3, "init_params": "kmeans"}


def fit_predict(X, params=None, random_state=42, threads=1):
    values = validate_matrix(X)
    settings = {**DEFAULTS, **(params or {})}
    validate_settings(settings, random_state, threads, "n_components", len(values))
    if settings.get("warm_start", False):
        raise ValueError("warm_start must remain False for independent experiment runs")
    if not isinstance(settings["n_init"], int) or isinstance(settings["n_init"], bool) or settings["n_init"] < 1:
        raise ValueError("n_init must be an explicit positive integer")
    model = GaussianMixture(**settings, random_state=random_state)
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=threads):
        warnings.simplefilter("always")
        start = time.perf_counter()
        model.fit(values)
        seconds = time.perf_counter() - start
        probabilities = model.predict_proba(values)
        labels = probabilities.argmax(axis=1)
        diagnostics = {"aic": float(model.aic(values)), "bic": float(model.bic(values)),
                       "lower_bound": float(model.lower_bound_), "weights": model.weights_.tolist(),
                       "distinct_clusters": int(len(np.unique(labels))), "threads": threads}
    status = ("not_converged" if not model.converged_ else
              "collapsed_clusters" if diagnostics["distinct_clusters"] < settings["n_components"] else "ok")
    return ClusteringResult(
        algorithm="gmm", labels=labels, centers=model.means_.copy(), fit_seconds=seconds,
        params=model.get_params(), seed=random_state, status=status, n_iter=int(model.n_iter_),
        converged=bool(model.converged_), warnings=[str(item.message) for item in captured],
        probabilities=probabilities, diagnostics=diagnostics,
    )
