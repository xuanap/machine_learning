"""Auditable cleaning of positive Online Retail purchase records."""
from __future__ import annotations

import numpy as np
import pandas as pd

SOURCE_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
    "UnitPrice", "CustomerID", "Country",
]
IDENTIFIERS = ["InvoiceNo", "StockCode", "CustomerID"]


def read_transactions(path):
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, dtype={c: "string" for c in IDENTIFIERS}, engine="openpyxl")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={c: "string" for c in IDENTIFIERS}, encoding="utf-8-sig")
    raise ValueError("Input must be .xlsx or .csv")


def clean_transactions(raw: pd.DataFrame, duplicate_policy: str = "drop_exact"):
    """Return cleaned data, rejected original rows, and an exclusive rule ledger.

    Exact duplicates refer to all eight ORIGINAL source columns. With no unique
    line ID, deduplication is a reproducible analysis assumption, not proof of error.
    """
    if duplicate_policy not in {"drop_exact", "keep"}:
        raise ValueError("duplicate_policy must be drop_exact or keep")
    missing = sorted(set(SOURCE_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing source columns: {missing}")
    original = raw[SOURCE_COLUMNS].reset_index(drop=True).copy()
    work = original.copy()
    work.insert(0, "SourceRow", np.arange(2, len(work) + 2))
    for column in IDENTIFIERS:
        work[column] = work[column].astype("string").str.strip().replace("", pd.NA)
    work["CustomerID"] = work["CustomerID"].str.replace(r"^([0-9]+)\.0+$", r"\1", regex=True)
    work["InvoiceDate"] = pd.to_datetime(work["InvoiceDate"], errors="coerce", format="mixed")
    for column in ["Quantity", "UnitPrice"]:
        work[column] = pd.to_numeric(work[column], errors="coerce").astype("float64")

    reason = pd.Series("", index=work.index, dtype="string")
    ledger = []

    def reject(rule, mask):
        newly_rejected = reason.eq("") & mask.fillna(False)
        before = int(reason.eq("").sum())
        reason.loc[newly_rejected] = rule
        ledger.append({"rule": rule, "rows_before": before,
                       "removed_rows": int(newly_rejected.sum()),
                       "rows_after": int(reason.eq("").sum())})

    reject("missing_customer_id", work["CustomerID"].isna())
    reject("invalid_customer_id", ~work["CustomerID"].str.fullmatch(r"[0-9]+", na=False))
    reject("missing_invoice_or_stock_code", work[["InvoiceNo", "StockCode"]].isna().any(axis=1))
    reject("invalid_invoice_date", work["InvoiceDate"].isna())
    reject("invalid_quantity_or_price", ~np.isfinite(work[["Quantity", "UnitPrice"]]).all(axis=1))
    reject("cancelled_invoice", work["InvoiceNo"].str.upper().str.startswith("C", na=False))
    reject("nonpositive_quantity", work["Quantity"].le(0))
    reject("nonpositive_unit_price", work["UnitPrice"].le(0))
    with np.errstate(over="ignore", invalid="ignore"):
        work["LineAmount"] = work["Quantity"] * work["UnitPrice"]
    reject("nonfinite_line_amount", ~np.isfinite(work["LineAmount"]))
    duplicate_mask = original.duplicated(subset=SOURCE_COLUMNS, keep="first")
    reject("exact_duplicate", duplicate_mask if duplicate_policy == "drop_exact" else pd.Series(False, index=work.index))

    cleaned = work.loc[reason.eq("")].copy().reset_index(drop=True)
    rejected = original.loc[reason.ne("")].copy()
    rejected.insert(0, "SourceRow", np.flatnonzero(reason.ne("")) + 2)
    rejected["ExclusionReason"] = reason.loc[reason.ne("")].to_numpy()
    return cleaned, rejected.reset_index(drop=True), pd.DataFrame(ledger)
