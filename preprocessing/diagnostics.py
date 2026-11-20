"""Return/cancellation diagnostics; these do not change the shared six features."""
import pandas as pd

from .data_cleaning import SOURCE_COLUMNS


def return_diagnostics(raw, cleaned, features, duplicate_policy):
    work = raw.copy().reset_index(drop=True)
    work.insert(0, "SourceRow", range(2, len(work) + 2))
    if duplicate_policy == "drop_exact":
        work = work.loc[~raw[SOURCE_COLUMNS].reset_index(drop=True).duplicated(keep="first")].copy()
    work["CustomerID"] = work["CustomerID"].astype("string").str.strip().str.replace(r"^([0-9]+)\.0+$", r"\1", regex=True)
    work["StockCode"] = work["StockCode"].astype("string").str.strip()
    work["InvoiceNo"] = work["InvoiceNo"].astype("string").str.strip()
    work["InvoiceDate"] = pd.to_datetime(work["InvoiceDate"], errors="coerce", format="mixed")
    for column in ["Quantity", "UnitPrice"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    cancelled = work.loc[work["InvoiceNo"].str.upper().str.startswith("C", na=False)
                         & work["CustomerID"].str.fullmatch(r"[0-9]+", na=False)].copy()
    cancelled["CancellationAmountAbs"] = (cancelled["Quantity"] * cancelled["UnitPrice"]).abs()
    grouped = cancelled.groupby("CustomerID", sort=True).agg(
        CancelledRows=("SourceRow", "size"), CancelledInvoices=("InvoiceNo", "nunique"),
        CancellationAmountAbs=("CancellationAmountAbs", "sum"))
    customer_summary = features[["CustomerID", "Monetary"]].merge(grouped, how="left", on="CustomerID")
    customer_summary[["CancelledRows", "CancelledInvoices"]] = customer_summary[["CancelledRows", "CancelledInvoices"]].fillna(0).astype("int64")
    customer_summary["CancellationAmountAbs"] = customer_summary["CancellationAmountAbs"].fillna(0)

    # Candidate links ONLY: equality is not proof of invoice-level return matching.
    largest = cleaned.nlargest(20, "LineAmount").copy()
    negative = cancelled.loc[cancelled["Quantity"].lt(0) & cancelled["UnitPrice"].gt(0)].copy()
    negative["Quantity"] = -negative["Quantity"]
    pairs = largest.merge(negative[["SourceRow", "CustomerID", "StockCode", "Quantity", "UnitPrice", "InvoiceNo", "InvoiceDate"]],
                          on=["CustomerID", "StockCode", "Quantity", "UnitPrice"], how="inner", suffixes=("", "_Cancellation"))
    pairs = pairs.loc[pairs["InvoiceDate_Cancellation"].ge(pairs["InvoiceDate"])].copy()
    return customer_summary, pairs
