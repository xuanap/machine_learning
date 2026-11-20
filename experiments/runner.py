"""One-factor experiments with traceable inputs and per-run artifacts."""
from __future__ import annotations

import json
import math
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from algorithms import gmm, kmeans
from preprocessing.feature_engineering import FEATURE_COLUMNS
from .inputs import ROOT, file_hash, matrix_hash
from .reference_metrics import evaluate

ALGORITHMS = {"kmeans": kmeans.fit_predict, "gmm": gmm.fit_predict}


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_tasks(config, suite):
    if suite not in {"baseline", "parameters", "ablation", "all"}:
        raise ValueError("Unknown suite")
    tasks = []

    def add(algorithm, study, factor, value, features, feature_set, **overrides):
        base = dict(config[algorithm])
        base["n_clusters" if algorithm == "kmeans" else "n_components"] = config["baseline_k"]
        base.update(overrides)
        for seed in config["seeds"]:
            tasks.append({"algorithm": algorithm, "study": study, "factor": factor, "value": value,
                          "features": features, "feature_set": feature_set, "params": base.copy(), "seed": seed})

    if suite in {"baseline", "all"}:
        for algorithm in ALGORITHMS:
            add(algorithm, "baseline", "algorithm", algorithm, FEATURE_COLUMNS, "six_features")
    if suite in {"parameters", "all"}:
        for count in config["k_values"]:
            add("kmeans", "kmeans_k", "K", count, FEATURE_COLUMNS, "six_features", n_clusters=count)
            add("gmm", "gmm_k", "K", count, FEATURE_COLUMNS, "six_features", n_components=count)
        for init in config["kmeans_initializations"]:
            add("kmeans", "kmeans_initialization", "init", init, FEATURE_COLUMNS, "six_features", init=init)
        for covariance in config["gmm_covariance_types"]:
            add("gmm", "gmm_covariance", "covariance_type", covariance, FEATURE_COLUMNS, "six_features", covariance_type=covariance)
    if suite in {"ablation", "all"}:
        for feature_set, columns in config["feature_sets"].items():
            for algorithm in ALGORITHMS:
                add(algorithm, "feature_ablation", "feature_set", feature_set, columns, feature_set)
    if not tasks:
        raise ValueError("Experiment configuration produced no tasks")
    return tasks


def run_suite(bundle, config, output, suite="baseline", metric_fn=evaluate,
              metrics_provider="A_reference_pending_C", make_plots=True):
    tasks = make_tasks(config, suite)
    for task in tasks:
        bundle.matrix(task["features"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model_input = output / "model_input.csv"
    bundle.frame.to_csv(model_input, index=False, encoding="utf-8-sig", float_format="%.17g", lineterminator="\n")
    saved = pd.read_csv(model_input, dtype={"CustomerID": "string"}, float_precision="round_trip")
    if matrix_hash(saved) != matrix_hash(bundle.frame):
        raise ValueError("Saved model coordinates do not round-trip exactly")
    code_files = [ROOT / "main.py", *sorted((ROOT / "algorithms").glob("*.py")),
                  *sorted((ROOT / "experiments").glob("*.py")), ROOT / "preprocessing/feature_scaling.py"]
    write_json(output / "experiment_manifest.json", {
        "mode": bundle.mode, "suite": suite, "input": bundle.provenance, "config": config,
        "model_input": {"filename": model_input.name, "sha256": file_hash(model_input),
                        "matrix_sha256": matrix_hash(saved)},
        "metrics_provider": metrics_provider,
        "timing": "estimator.fit only, including estimator initialization; excludes prediction, metrics, file I/O and plotting",
        "threads": config["threads"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
        "code_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): file_hash(path) for path in code_files},
    })
    bundle.frame[["CustomerID"]].to_csv(output / "sample_customer_ids.csv", index=False, encoding="utf-8-sig")
    summaries = []
    for index, task in enumerate(tasks):
        run_id = f"{index:03d}_{task['algorithm']}_seed{task['seed']}"
        folder = output / run_id
        folder.mkdir()
        values = bundle.matrix(task["features"])
        record = {"run_id": run_id, "mode": bundle.mode, "algorithm": task["algorithm"],
                  "study": task["study"], "factor": task["factor"], "value": task["value"],
                  "feature_set": task["feature_set"], "seed": task["seed"],
                  "samples": len(values), "features": len(task["features"]),
                  "requested_params": json.dumps(task["params"], sort_keys=True)}
        try:
            result = ALGORITHMS[task["algorithm"]](values, task["params"], task["seed"], config["threads"])
        except (ValueError, TypeError, np.linalg.LinAlgError) as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(folder / "run.json", {"task": task, "summary": record})
            summaries.append(record)
            continue
        labels = bundle.frame[["CustomerID"]].copy()
        labels["Cluster"] = result.labels
        labels.to_csv(folder / "labels.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(result.centers, columns=task["features"]).rename_axis("Cluster").to_csv(folder / "centers.csv", encoding="utf-8-sig")
        labels.groupby("Cluster", sort=True).size().rename("Customers").to_csv(folder / "cluster_sizes.csv", encoding="utf-8-sig")
        if result.probabilities is not None:
            probabilities = bundle.frame[["CustomerID"]].copy()
            for cluster in range(result.probabilities.shape[1]):
                probabilities[f"Probability_{cluster}"] = result.probabilities[:, cluster]
            probabilities.to_csv(folder / "probabilities.csv", index=False, encoding="utf-8-sig")
        record.update(status=result.status, fit_seconds=result.fit_seconds, n_iter=result.n_iter,
                      converged=result.converged, warning_count=len(result.warnings), **result.diagnostics)
        try:
            record.update(metric_fn(values, result.labels))
        except (ValueError, TypeError, np.linalg.LinAlgError) as error:
            record.update(status="metric_error", error=f"{type(error).__name__}: {error}")
        details = asdict(result)
        for field in ["labels", "centers", "probabilities"]:
            details.pop(field)
        write_json(folder / "run.json", {"task": task, "fit": details, "summary": record})
        summaries.append(record)
    table = pd.DataFrame(summaries)
    table.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    if make_plots:
        from .plots import plot_summary
        plot_summary(table, output / "figures", bundle.mode)
    (output / "NOTICE.txt").write_text(
        "SMOKE TEST ONLY. Raw features; small diagnostic sample. Not a formal comparison or evidence of best parameters.\n"
        if bundle.mode == "smoke" else
        f"Full-customer prepared-input study. Metric provider: {metrics_provider}. Group approval and timing comparability must not be inferred from this run.\n",
        encoding="utf-8")
    return table
