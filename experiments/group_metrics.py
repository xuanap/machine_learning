"""Full-data metrics matching C's current spectral entrypoint, not a team approval."""
import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from threadpoolctl import threadpool_limits


def evaluate_full(X, labels):
    values = np.asarray(X)
    labels = np.asarray(labels)
    if not 2 <= len(np.unique(labels)) < len(labels):
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None,
                "silhouette_samples": 0, "metric_note": "Metrics need 2..n_samples-1 clusters"}
    # Thread limits affect cost, not the definition: all customers, Euclidean distance.
    with threadpool_limits(limits=1):
        return {"silhouette": float(silhouette_score(values, labels)),
                "davies_bouldin": float(davies_bouldin_score(values, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(values, labels)),
                "silhouette_samples": len(labels), "metric_note": ""}


def evaluate_comparison(X, labels):
    """Score assigned customers and report the coverage of those scores."""
    values = np.asarray(X)
    labels = np.asarray(labels)
    assigned = labels != -1
    coverage = float(assigned.mean())
    return {
        **evaluate_full(values[assigned], labels[assigned]),
        "observed_clusters": len(np.unique(labels[assigned])),
        "coverage": coverage,
        "noise_count": int((~assigned).sum()),
    }
