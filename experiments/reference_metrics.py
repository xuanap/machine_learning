"""Local reference metrics for testing; C owns the group's final metric protocol."""
import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def evaluate(X, labels, sample_size=1000, random_state=2026):
    values = np.asarray(X)
    labels = np.asarray(labels)
    result = {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None,
              "silhouette_samples": 0, "metric_note": ""}
    if not 2 <= len(np.unique(labels)) < len(labels):
        result["metric_note"] = "Internal metrics require 2..n_samples-1 distinct clusters"
        return result
    result["davies_bouldin"] = float(davies_bouldin_score(values, labels))
    result["calinski_harabasz"] = float(calinski_harabasz_score(values, labels))
    limit = min(sample_size, len(labels))
    try:
        result["silhouette"] = float(silhouette_score(values, labels, sample_size=limit, random_state=random_state))
        result["silhouette_samples"] = limit
    except ValueError as error:
        result["metric_note"] = str(error)
    return result
