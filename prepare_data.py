"""Build the shared B/C data delivery; run from any working directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from preprocessing.data_cleaning import SOURCE_COLUMNS, clean_transactions, read_transactions
from preprocessing.feature_engineering import FEATURE_COLUMNS, build_customer_features
from preprocessing.diagnostics import return_diagnostics

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/data_config.json")
    parser.add_argument("--input", type=Path, help="Override workbook path")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["scope"] != "all_countries_positive_purchases" or config["transform"] != "none":
        raise ValueError("This pipeline implements only all-country positive purchases without feature transforms")
    source = args.input.resolve() if args.input else ROOT / config["input"]
    print(f"Reading {source.name}", flush=True)
    raw = read_transactions(source)
    raw_dates = pd.to_datetime(raw["InvoiceDate"], errors="coerce", format="mixed")
    raw_profile = {
        "rows": len(raw), "columns": list(raw.columns),
        "missing_values": {column: int(raw[column].isna().sum()) for column in raw.columns},
        "exact_duplicate_rows_beyond_first": int(raw.duplicated(SOURCE_COLUMNS).sum()),
        "distinct_customer_ids_nonmissing": int(raw["CustomerID"].nunique()),
        "invoice_date_min": str(raw_dates.min()), "invoice_date_max": str(raw_dates.max()),
        "cancelled_rows_before_other_filters": int(raw["InvoiceNo"].astype("string").str.strip().str.upper().str.startswith("C", na=False).sum()),
    }
    print(f"Raw rows: {len(raw):,}; cleaning...", flush=True)
    cleaned, excluded, ledger = clean_transactions(raw, config["duplicate_policy"])
    features, dates, reference = build_customer_features(cleaned, config["analysis_date"])

    data_dir = ROOT / "data"
    tables = ROOT / "results/tables"
    data_dir.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    csv_options = {"index": False, "encoding": "utf-8-sig", "lineterminator": "\n"}
    cleaned.to_csv(data_dir / "cleaned_data.csv", date_format="%Y-%m-%d %H:%M:%S", **csv_options)
    features.to_csv(data_dir / "customer_features.csv", **csv_options)
    dates.to_csv(data_dir / "customer_dates.csv", date_format="%Y-%m-%d", **csv_options)
    excluded.to_csv(data_dir / "excluded_rows.csv.gz", compression={"method": "gzip", "mtime": 0}, **csv_options)
    ledger.to_csv(tables / "cleaning_steps.csv", **csv_options)
    features[FEATURE_COLUMNS].describe(percentiles=[.25, .5, .75, .95, .99]).T.to_csv(
        tables / "feature_summary.csv", index_label="Feature", encoding="utf-8-sig", lineterminator="\n")
    non_numeric = ~cleaned["StockCode"].str.fullmatch(r"[0-9]+[A-Za-z]?", na=False)
    diagnostics = cleaned.loc[non_numeric].groupby("StockCode", sort=True).agg(
        Rows=("SourceRow", "size"), Customers=("CustomerID", "nunique"), TotalAmount=("LineAmount", "sum"))
    diagnostics.to_csv(tables / "special_stock_codes.csv", encoding="utf-8-sig", lineterminator="\n")
    for column in ["Quantity", "UnitPrice", "LineAmount"]:
        cleaned.nlargest(20, column).to_csv(tables / f"largest_{column.lower()}.csv", **csv_options)
    cancellations, candidate_pairs = return_diagnostics(raw, cleaned, features, config["duplicate_policy"])
    cancellations.to_csv(tables / "customer_cancellation_diagnostics.csv", **csv_options)
    candidate_pairs.to_csv(tables / "largest_order_cancellation_candidates.csv", date_format="%Y-%m-%d %H:%M:%S", **csv_options)
    inconsistent = cleaned.groupby("InvoiceNo")["CustomerID"].nunique()
    invoice_conflicts = int(inconsistent.gt(1).sum())
    # These are validation gates, not a substitute for the independent verifier.
    checks = {
        "row_count_reconciles": len(raw) == len(cleaned) + len(excluded),
        "ledger_reconciles": int(ledger["removed_rows"].sum()) == len(excluded),
        "customer_id_unique": not features["CustomerID"].duplicated().any(),
        "features_finite": bool(np.isfinite(features[FEATURE_COLUMNS]).all().all()),
        "feature_customer_coverage": len(features) == cleaned["CustomerID"].nunique(),
        "monetary_reconciles": bool(np.isclose(features["Monetary"].sum(), cleaned["LineAmount"].sum(), rtol=0, atol=1e-6)),
        "valid_feature_ranges": bool(features["Recency"].ge(1).all() and features["ActiveDays"].ge(0).all()
                                     and features[["Frequency", "Monetary", "AvgOrderValue", "ProductDiversity"]].gt(0).all().all()),
        "invoice_has_one_customer": invoice_conflicts == 0,
    }
    save_json(tables / "raw_profile.json", raw_profile)
    save_json(tables / "pipeline_checks.json", {key: bool(value) for key, value in checks.items()})
    if not all(checks.values()):
        raise ValueError(f"Validation failed: {[key for key, value in checks.items() if not value]}")
    outputs = [data_dir / name for name in ["cleaned_data.csv", "customer_features.csv", "customer_dates.csv", "excluded_rows.csv.gz"]]
    manifest = {
        "data_version": config["data_version"], "config": config,
        "input_filename": source.name, "input_sha256": sha256(source),
        "analysis_date": reference.strftime("%Y-%m-%d"),
        "purchase_date_min": str(cleaned["InvoiceDate"].min()),
        "purchase_date_max": str(cleaned["InvoiceDate"].max()),
        "raw_rows": len(raw), "cleaned_rows": len(cleaned), "excluded_rows": len(excluded),
        "customers": len(features), "invoices": int(cleaned["InvoiceNo"].nunique()),
        "total_monetary_gbp": float(features["Monetary"].sum()),
        "feature_columns": FEATURE_COLUMNS, "identifier_column": "CustomerID",
        "customer_order": "CustomerID ascending as text",
        "scope_notes": ["all countries", "positive purchases; refunds are not netted against original purchases",
                        "exact original eight-column duplicates removed unless duplicate_policy=keep",
                        "no monetary outlier trimming", "no log1p, standardization, or PCA",
                        "non-product and service StockCode values retained; ProductDiversity counts all retained codes"],
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__, "openpyxl": openpyxl.__version__},
        "code_sha256": {str(path.relative_to(ROOT)).replace('\\', '/'): sha256(path) for path in
                        [Path(__file__), ROOT / "preprocessing/data_cleaning.py", ROOT / "preprocessing/feature_engineering.py", ROOT / "preprocessing/diagnostics.py"]},
        "files": {str(path.relative_to(ROOT)).replace('\\', '/'): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs},
    }
    save_json(data_dir / "dataset_manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ["data_version", "raw_rows", "cleaned_rows", "excluded_rows", "customers", "invoices", "analysis_date", "total_monetary_gbp"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
