"""Independently check the saved CSVs using stdlib CSV, sets, dates, and Decimal.

No cleaning/feature implementation is imported. Every retained customer and every
input row's kept/excluded membership are checked, including a file-hash check.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = Decimal("0.000001")


def read_csv(path):
    return path.open(encoding="utf-8-sig", newline="")


def main():
    manifest = json.loads((ROOT / "data/dataset_manifest.json").read_text(encoding="utf-8"))
    reference = date.fromisoformat(manifest["analysis_date"])
    checks = []

    def check(name, valid):
        checks.append({"check": name, "passed": bool(valid)})
        if not valid:
            raise ValueError(f"Independent validation failed: {name}")

    for name, metadata in manifest["files"].items():
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        check(f"sha256:{name}", digest == metadata["sha256"])

    aggregates = defaultdict(lambda: {"orders": set(), "products": set(), "days": set(), "amount": Decimal(0)})
    kept_rows = set()
    invoice_owners = {}
    count = 0
    all_positive = True
    all_amounts_match = True
    with read_csv(ROOT / "data/cleaned_data.csv") as stream:
        for row in csv.DictReader(stream):
            count += 1
            row_id = int(row["SourceRow"])
            if row_id in kept_rows:
                raise ValueError("Repeated SourceRow in cleaned data")
            kept_rows.add(row_id)
            quantity = Decimal(row["Quantity"])
            price = Decimal(row["UnitPrice"])
            amount = quantity * price
            all_positive &= quantity.is_finite() and price.is_finite() and quantity > 0 and price > 0
            all_amounts_match &= abs(amount - Decimal(row["LineAmount"])) <= TOLERANCE
            if row["InvoiceNo"].upper().startswith("C"):
                raise ValueError("Cancelled invoice retained")
            owner = invoice_owners.setdefault(row["InvoiceNo"], row["CustomerID"])
            if owner != row["CustomerID"]:
                raise ValueError("Invoice has conflicting customer IDs")
            item = aggregates[row["CustomerID"]]
            item["orders"].add(row["InvoiceNo"])
            item["products"].add(row["StockCode"])
            item["days"].add(datetime.fromisoformat(row["InvoiceDate"]).date())
            item["amount"] += amount
    check("all_transactions_positive_and_finite", all_positive)
    check("line_amounts_recomputed_with_decimal", all_amounts_match)
    check("cleaned_row_count", count == manifest["cleaned_rows"])

    excluded_rows = set()
    with gzip.open(ROOT / "data/excluded_rows.csv.gz", "rt", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            row_id = int(row["SourceRow"])
            if row_id in excluded_rows or not row["ExclusionReason"]:
                raise ValueError("Duplicate exclusion or missing reason")
            excluded_rows.add(row_id)
    check("excluded_row_count", len(excluded_rows) == manifest["excluded_rows"])
    check("kept_excluded_disjoint", kept_rows.isdisjoint(excluded_rows))
    check("every_source_row_accounted_for", kept_rows | excluded_rows == set(range(2, manifest["raw_rows"] + 2)))

    customer_ids = []
    feature_monetary_total = Decimal(0)
    sample_details = []
    with read_csv(ROOT / "data/customer_features.csv") as stream:
        reader = csv.DictReader(stream)
        check("exact_feature_schema", reader.fieldnames == ["CustomerID"] + manifest["feature_columns"])
        for row in reader:
            customer = row["CustomerID"]
            customer_ids.append(customer)
            item = aggregates[customer]
            expected = {
                "Recency": (reference - max(item["days"])).days,
                "Frequency": len(item["orders"]),
                "Monetary": item["amount"],
                "AvgOrderValue": item["amount"] / len(item["orders"]),
                "ProductDiversity": len(item["products"]),
                "ActiveDays": (max(item["days"]) - min(item["days"])).days,
            }
            for column, value in expected.items():
                if abs(Decimal(row[column]) - Decimal(value)) > TOLERANCE:
                    raise ValueError(f"Customer {customer}: {column} does not match independent recomputation")
            feature_monetary_total += Decimal(row["Monetary"])
            if len(sample_details) < 5:
                sample_details.append({"CustomerID": customer, **{key: str(value) for key, value in expected.items()}})
    check("every_customer_and_all_six_features_recomputed", len(customer_ids) == len(aggregates) == manifest["customers"] and set(customer_ids) == set(aggregates))
    check("unique_sorted_customer_order", customer_ids == sorted(set(customer_ids)))
    exact_total = sum((item["amount"] for item in aggregates.values()), Decimal(0))
    check("total_monetary_reconciles", abs(exact_total - feature_monetary_total) <= TOLERANCE)
    check("manifest_monetary_reconciles", abs(exact_total - Decimal(str(manifest["total_monetary_gbp"]))) <= TOLERANCE)
    report = {"status": "passed", "checks": checks, "checked_customers": len(customer_ids),
              "checked_transactions": count, "exact_total_monetary_gbp": str(exact_total),
              "tolerance_gbp": str(TOLERANCE), "sample_recalculations": sample_details}
    (ROOT / "results/tables/independent_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["status", "checked_customers", "checked_transactions", "exact_total_monetary_gbp"]}, indent=2), flush=True)
    print(f"{len(checks)} independent checks passed", flush=True)


if __name__ == "__main__":
    main()
