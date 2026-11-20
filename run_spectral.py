"""Run spectral clustering using the existing six customer features."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from algorithms.spectral import spectral_clustering
from preprocessing.feature_engineering import FEATURE_COLUMNS

LOG_COLUMNS = ["Frequency", "Monetary", "AvgOrderValue", "ProductDiversity"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "data" / "customer_features.csv")
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--affinity", choices=["nearest_neighbors", "rbf"], default="nearest_neighbors")
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optional CSV path for CustomerID and Cluster.")
    args = parser.parse_args()

    customers = pd.read_csv(args.input, dtype={"CustomerID": str})
    ids = customers["CustomerID"]
    if ids.isna().any() or not ids.is_unique:
        parser.error("CustomerID must be present and unique for each row.")

    features = customers[FEATURE_COLUMNS].astype(float)
    if not np.isfinite(features.to_numpy()).all() or (features < 0).any().any():
        parser.error("Customer features must be finite and nonnegative.")
    features[LOG_COLUMNS] = np.log1p(features[LOG_COLUMNS])
    X = StandardScaler().fit_transform(features)

    start = perf_counter()
    labels = spectral_clustering(
        X, n_clusters=args.n_clusters, affinity=args.affinity,
        n_neighbors=args.n_neighbors, gamma=args.gamma, random_state=args.random_state,
    )
    elapsed = perf_counter() - start
    cluster_ids, counts = np.unique(labels, return_counts=True)
    summary = {
        "customers": len(labels),
        "cluster_sizes": dict(zip(cluster_ids.astype(str).tolist(), counts.tolist())),
        "runtime_seconds": round(elapsed, 4),
    }
    if 2 <= len(counts) < len(labels):
        summary.update(
            silhouette=float(silhouette_score(X, labels)),
            davies_bouldin=float(davies_bouldin_score(X, labels)),
            calinski_harabasz=float(calinski_harabasz_score(X, labels)),
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"CustomerID": ids, "Cluster": labels}).to_csv(args.output, index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
