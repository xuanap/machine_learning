"""Compare two saved runs only when data, customers and feature spaces agree."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from analysis.compare_clusters import compare_partitions
from experiments.inputs import ROOT, file_hash
from experiments.runner import write_json


def read_run(experiment, run_id):
    experiment = Path(experiment)
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a direct run directory name")
    manifest_path = experiment / "experiment_manifest.json"
    run_path = experiment / run_id / "run.json"
    labels_path = experiment / run_id / "labels.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if manifest["mode"] not in {"smoke", "formal"} or run["summary"]["mode"] != manifest["mode"]:
        raise ValueError("Invalid or inconsistent run mode")
    if run["summary"]["status"] != "ok":
        raise ValueError("Only status=ok partitions can be compared")
    labels = pd.read_csv(labels_path, dtype={"CustomerID": "string"})
    expected = pd.read_csv(experiment / "sample_customer_ids.csv", dtype={"CustomerID": "string"})
    if len(labels) != run["summary"]["samples"] or len(expected) != len(labels) or set(labels.CustomerID) != set(expected.CustomerID):
        raise ValueError("Saved labels do not cover the recorded experiment customers")
    provenance = {"experiment": experiment.name, "run_id": run_id,
                  "manifest_sha256": file_hash(manifest_path), "run_sha256": file_hash(run_path),
                  "labels_sha256": file_hash(labels_path)}
    return manifest, run, labels, provenance


def compare_runs(left_experiment, left_run, right_experiment, right_run, output):
    lm, lr, left, lp = read_run(left_experiment, left_run)
    rm, rr, right, rp = read_run(right_experiment, right_run)
    if lm["mode"] != rm["mode"]:
        raise ValueError("Cannot mix smoke and formal runs")
    for field in ["data_version", "source_features_sha256", "input_sha256", "preprocessing"]:
        if lm["input"][field] != rm["input"][field]:
            raise ValueError(f"Runs have different {field}")
    if lm["input"].get("matrix_sha256") != rm["input"].get("matrix_sha256"):
        raise ValueError("Runs have different model coordinates")
    if lr["task"]["features"] != rr["task"]["features"]:
        raise ValueError("Use the same ordered features for this algorithm comparison")
    customers, overlap, fractions, mapping, summary = compare_partitions(left, right)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    customers.to_csv(output / "customer_comparison.csv", index=False, encoding="utf-8-sig")
    overlap.to_csv(output / "overlap_counts.csv", encoding="utf-8-sig")
    fractions.to_csv(output / "overlap_row_fractions.csv", encoding="utf-8-sig")
    mapping.to_csv(output / "cluster_mapping.csv", index=False, encoding="utf-8-sig")
    customers.loc[~customers.SameAfterMatching].to_csv(output / "different_assignments.csv", index=False, encoding="utf-8-sig")
    write_json(output / "comparison.json", {"mode": lm["mode"], **summary, "left": lp, "right": rp,
               "left_algorithm": lr["summary"]["algorithm"], "right_algorithm": rr["summary"]["algorithm"],
               "input": lm["input"], "features": lr["task"]["features"],
               "code_sha256": {"compare_runs.py": file_hash(Path(__file__)),
                               "analysis/compare_clusters.py": file_hash(ROOT / "analysis/compare_clusters.py")}})
    (output / "NOTICE.txt").write_text(
        "SMOKE ONLY: diagnostic partition comparison, not formal customer behavior or business conclusions.\n"
        if lm["mode"] == "smoke" else
        "Partition consistency only. Matched overlap is not classification accuracy or a real behavior change rate.\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-experiment", type=Path, required=True)
    parser.add_argument("--left-run", required=True)
    parser.add_argument("--right-experiment", type=Path)
    parser.add_argument("--right-run", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    right_experiment = args.right_experiment or args.left_experiment
    output = args.output or ROOT / "results/comparison" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        summary = compare_runs(args.left_experiment, args.left_run, right_experiment, args.right_run, output)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
