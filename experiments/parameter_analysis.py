"""Run one-factor spectral clustering experiments on the shared formal input."""

import argparse
import platform
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse.linalg import ArpackNoConvergence
from threadpoolctl import threadpool_limits

from algorithms.spectral import spectral_clustering
from experiments.group_metrics import evaluate_full
from experiments.inputs import ROOT, file_hash, load_input
from experiments.runner import write_json


NEIGHBOR_COUNTS = (5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300, 400, 600)

def designs():
    knn = {"affinity": "nearest_neighbors", "n_neighbors": 20}
    rbf = {"affinity": "rbf", "gamma": 0.1}
    return {
        "k": [dict(base, n_clusters=k) for base in (knn, rbf) for k in range(2, 11)],
        "affinity": [dict(base, n_clusters=2) for base in (knn, rbf)],
        "n_neighbors": [
            dict(knn, n_clusters=2, n_neighbors=n) for n in NEIGHBOR_COUNTS
        ],
        "gamma": [
            dict(rbf, n_clusters=2, gamma=g) for g in (0.01, 0.025, 0.05, 0.1, 0.2, 0.5)
        ],
    }


def run_case(X, params, seed=42):
    row = dict(params, random_state=seed, customers=len(X), status="ok")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        start = perf_counter()
        try:
            labels = spectral_clustering(X, **params, random_state=seed)
        except (ValueError, ArpackNoConvergence) as error:
            return dict(row, status="failed", note=str(error))
        row["runtime_seconds"] = perf_counter() - start
    counts = sorted(np.unique(labels, return_counts=True)[1])
    return dict(
        row, **evaluate_full(X, labels), observed_clusters=len(counts),
        smallest_cluster_share=counts[0] / len(X),
        cluster_sizes_ascending="|".join(map(str, counts)),
        note=" | ".join(str(entry.message) for entry in captured),
    )


def main():
    studies = designs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=["all", *studies, "n_neighbors_seeds"], default="all")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/tables/spectral_parameters")
    args = parser.parse_args()
    bundle = load_input("formal")
    X = bundle.matrix(bundle.provenance["feature_columns"])
    if args.step == "n_neighbors_seeds":
        studies[args.step] = [case for case in studies["n_neighbors"] if case["n_neighbors"] >= 150]
    selected = studies if args.step == "all" else {args.step: studies[args.step]}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with threadpool_limits(limits=1):
        for step, cases in selected.items():
            seeds = (0, 42, 2026) if step == "n_neighbors_seeds" else (42,)
            rows = [run_case(X, params, seed) for params in cases for seed in seeds]
            path = args.output_dir / f"{step}.csv"
            pd.DataFrame(rows).to_csv(path, index=False, float_format="%.8f")
            write_json(path.with_suffix(".json"), {
                "step": step, "input": bundle.provenance, "cases": cases,
                "random_state": seeds[0] if len(seeds) == 1 else seeds, "threads": 1,
                "timing": "spectral_clustering only",
                "csv_sha256": file_hash(path),
                "code_sha256": {
                    name: file_hash(ROOT / name) for name in (
                        "experiments/parameter_analysis.py", "experiments/group_metrics.py",
                        "algorithms/spectral.py", "experiments/inputs.py",
                    )
                },
                "environment": {
                    "python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scipy": scipy.__version__,
                    "scikit_learn": sklearn.__version__,
                },
            })
            print(f"{step}: {sum(row['status'] == 'ok' for row in rows)}/{len(rows)} completed → {path}")


if __name__ == "__main__":
    main()
