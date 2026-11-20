"""Analyze one saved experiment using raw-scale profiles and fitted-space boundaries."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from analysis.customer_profiles import build_profiles, gmm_boundaries, kmeans_boundaries
from experiments.inputs import ROOT, file_hash
from experiments.runner import write_json


def analyze_run(experiment, run_id, output, matrix_input=None):
    experiment = Path(experiment)
    manifest_path = experiment / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["mode"] not in {"smoke", "formal"}:
        raise ValueError("Unknown experiment mode")
    if not isinstance(run_id, str) or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must name a direct run directory")
    run = experiment / run_id
    run_path = run / "run.json"
    record = json.loads(run_path.read_text(encoding="utf-8"))
    if record["summary"]["status"] != "ok":
        raise ValueError("Only status=ok runs are accepted for customer analysis")
    if record["summary"]["mode"] != manifest["mode"]:
        raise ValueError("Run and experiment mode disagree")
    raw_path = ROOT / "data/customer_features.csv"
    if file_hash(raw_path) != manifest["input"]["source_features_sha256"]:
        raise ValueError("Original features do not match this experiment's source version")
    raw = pd.read_csv(raw_path, dtype={"CustomerID": "string"})
    label_path = run / "labels.csv"
    labels = pd.read_csv(label_path, dtype={"CustomerID": "string"})
    if len(labels) != record["summary"]["samples"]:
        raise ValueError("Label count differs from the saved run")
    expected_ids = pd.read_csv(experiment / "sample_customer_ids.csv", dtype={"CustomerID": "string"})
    if expected_ids.CustomerID.duplicated().any() or set(labels.CustomerID) != set(expected_ids.CustomerID):
        raise ValueError("Label customers differ from experiment customers")
    assignments, summary, statistics = build_profiles(raw, labels)
    algorithm = record["summary"]["algorithm"]
    artifacts = {"experiment_manifest.json": file_hash(manifest_path), "run.json": file_hash(run_path),
                 "labels.csv": file_hash(label_path)}
    if algorithm == "gmm":
        path = run / "probabilities.csv"
        probabilities = pd.read_csv(path, dtype={"CustomerID": "string"})
        assignments = gmm_boundaries(assignments, probabilities)
        artifacts["probabilities.csv"] = file_hash(path)
        boundary_column = "ProbabilityGap"
        representative_column, ascending = "HighestProbability", False
    elif algorithm == "kmeans":
        path = Path(matrix_input) if matrix_input is not None else ROOT / "data" / manifest["input"]["input_filename"]
        if not path.exists():
            raise ValueError("The original model-input CSV is needed; pass --matrix-input without redoing preprocessing")
        if file_hash(path) != manifest["input"]["input_sha256"]:
            raise ValueError("Model-input CSV differs from the matrix used to fit this run")
        matrix = pd.read_csv(path, dtype={"CustomerID": "string"})
        centers_path = run / "centers.csv"
        centers = pd.read_csv(centers_path)
        assignments = kmeans_boundaries(assignments, matrix, centers, record["task"]["features"])
        artifacts.update({"model_input.csv": file_hash(path), "centers.csv": file_hash(centers_path)})
        boundary_column = "DistanceGap"
        representative_column, ascending = "AssignedCenterDistance", True
    else:
        raise ValueError("This analysis supports A's kmeans and gmm runs only")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    options = {"index": False, "encoding": "utf-8-sig"}
    assignments.to_csv(output / "customer_assignments.csv", **options)
    summary.to_csv(output / "cluster_summary.csv", **options)
    statistics.to_csv(output / "feature_statistics.csv", **options)
    assignments.loc[assignments[boundary_column].notna()].sort_values(boundary_column, kind="stable").head(10).to_csv(output / "boundary_customers.csv", **options)
    assignments.sort_values(representative_column, ascending=ascending, kind="stable").groupby("Cluster", sort=True).head(3).to_csv(output / "representative_customers.csv", **options)
    write_json(output / "analysis_manifest.json", {
        "mode": manifest["mode"], "algorithm": algorithm, "run_id": run_id,
        "data_version": manifest["input"]["data_version"], "customers": len(assignments),
        "source_customers": len(raw), "shares_denominator": "customers and monetary in this analysed run, not the entire source population",
        "original_features_sha256": file_hash(raw_path), "artifacts_sha256": artifacts,
        "code_sha256": {"analyze_run.py": file_hash(Path(__file__)),
                        "analysis/customer_profiles.py": file_hash(ROOT / "analysis/customer_profiles.py")},
        "note": "Profiles are in original units; distances are in the exact fitted feature space; Monetary is not net of returns",
    })
    (output / "NOTICE.txt").write_text(
        "SMOKE ONLY: diagnostic sample, not formal customer segments or evidence of business value.\n"
        if manifest["mode"] == "smoke" else
        "Original-scale descriptive profiles. Gross positive purchase amounts are not net of returns.\n",
        encoding="utf-8")
    return assignments, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--matrix-input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        manifest = json.loads((args.experiment / "experiment_manifest.json").read_text(encoding="utf-8"))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = args.output or ROOT / "results/analysis" / manifest["mode"] / f"{args.run_id}_{timestamp}"
        assignments, summary = analyze_run(args.experiment, args.run_id, output, args.matrix_input)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"Mode: {manifest['mode']}; customers: {len(assignments)}; clusters: {len(summary)}")
    print(f"Saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
