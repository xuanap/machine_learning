"""Entry point for B's EDA and two full-data clustering experiments."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

if sys.version_info < (3, 11) or sys.version_info.releaselevel != "final":
    raise SystemExit("B requires a stable Python >=3.11 release; preview builds can break binary dependencies.")

import numpy as np
import pandas as pd

from analysis.b_eda import run_eda
from experiments.b_report import write_report
from experiments.b_runner import EPS_GRID, MIN_SAMPLES_GRID, run_clustering
from preprocessing.feature_engineering import FEATURE_COLUMNS
from preprocessing.feature_scaling import LOG_COLUMNS

ROOT = Path(__file__).resolve().parent


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_environment():
    versions = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if "==" not in line or line.lstrip().startswith("#"):
            continue
        package, expected = line.strip().split("==")
        actual = version(package)
        if actual != expected:
            raise RuntimeError(f"{package}=={expected} required, found {actual}; install requirements.txt")
        versions[package] = actual
    return {"python": platform.python_version(), "python_executable": sys.executable,
            "python_base_prefix": sys.base_prefix, **versions}


def main():
    parser = argparse.ArgumentParser(description="B: full-data EDA, hierarchical clustering and DBSCAN")
    parser.add_argument("--mode", choices=["eda", "clustering", "all"], default="all")
    parser.add_argument("--output", type=Path, help="New directory inside results/b; default: timestamped directory")
    args = parser.parse_args()
    environment = check_environment()
    now = datetime.now(timezone(timedelta(hours=8)))
    output = (args.output if args.output is not None else ROOT / "results/b" / f"b_{now:%Y%m%d_%H%M%S_%f}").resolve()
    if not output.is_relative_to((ROOT / "results/b").resolve()) or output == (ROOT / "results/b").resolve():
        parser.error("--output must be a new subdirectory of results/b")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "mode": args.mode, "started_at_shanghai": now.isoformat(),
                "environment": environment, "feature_columns": FEATURE_COLUMNS,
                "preprocessing": {"function": "preprocessing.feature_scaling.load_customer_features",
                                  "log1p_columns": LOG_COLUMNS, "scaler": "StandardScaler", "pca": False},
                "threads": 1, "random_state": 42,
                "note": "Both algorithms deterministic; seed retained only for shared API compatibility.",
                "dbscan_grid": {"eps": EPS_GRID, "min_samples": MIN_SAMPLES_GRID}}
    sources = ["data/cleaned_data.csv", "data/customer_features.csv", "data/dataset_manifest.json",
               "results/tables/customer_cancellation_diagnostics.csv", "results/tables/largest_order_cancellation_candidates.csv",
               "requirements.txt", "preprocessing/feature_scaling.py", "preprocessing/feature_engineering.py",
               "algorithms/common.py", "algorithms/hierarchical.py", "algorithms/dbscan.py", "analysis/b_eda.py",
               "experiments/b_runner.py", "experiments/b_report.py", "experiments/group_metrics.py",
               "tests/test_b_work.py", "tools/verify_b_run.py", "run_b.py"]
    manifest["sources_sha256"] = {name: file_hash(ROOT / name) for name in sources}
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Output: {output}", flush=True)
    try:
        raw = pd.read_csv(ROOT / "data/customer_features.csv", dtype={"CustomerID": "string"})
        if list(raw.columns) != ["CustomerID", *FEATURE_COLUMNS] or raw.CustomerID.isna().any() or raw.CustomerID.duplicated().any():
            raise ValueError("Expected unique customer IDs and exactly the six agreed features")
        if not np.isfinite(raw[FEATURE_COLUMNS].to_numpy(dtype=float)).all() or raw[FEATURE_COLUMNS].lt(0).any().any():
            raise ValueError("Raw features must be finite and nonnegative")
        eda = clustering = None
        if args.mode in {"eda", "all"}:
            print("EDA: positive orders, monthly observation, concentration and six features.", flush=True)
            eda = run_eda(ROOT, output / "eda", raw)
            print(f"EDA done: {eda[0]['detail_rows']} lines, {eda[0]['orders']} orders.", flush=True)
        if args.mode in {"clustering", "all"}:
            clustering = run_clustering(ROOT, output / "clustering", raw, progress=lambda message: print(message, flush=True))
            X = clustering["matrix"]
            manifest["matrix"] = {"shape": list(X.shape), "sha256_float64_c_order": hashlib.sha256(X.tobytes(order="C")).hexdigest()}
            manifest["dbscan_selection"] = clustering["selection"]
        write_report(output / "experiment_report_zh.txt", eda, clustering)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["outputs_sha256"] = {str(item.relative_to(output)).replace("\\", "/"): file_hash(item)
                                       for item in sorted(output.rglob("*")) if item.is_file() and item != path}
        manifest["finished_at_shanghai"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed. Chinese report: {output / 'experiment_report_zh.txt'}", flush=True)


if __name__ == "__main__":
    main()
