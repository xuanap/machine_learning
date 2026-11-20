"""Six unscaled customer features; no model fitting or PCA is performed here."""
from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "ProductDiversity", "ActiveDays"]


def build_customer_features(cleaned: pd.DataFrame, analysis_date=None):
    if cleaned.empty:
        raise ValueError("No valid purchases remain")
    purchase_days = cleaned["InvoiceDate"].dt.normalize()
    reference = (purchase_days.max() + pd.Timedelta(days=1) if analysis_date is None
                 else pd.Timestamp(analysis_date).normalize())
    if reference <= purchase_days.max():
        raise ValueError("analysis_date must be after the last retained purchase day")
    data = cleaned.assign(PurchaseDay=purchase_days)
    grouped = data.groupby("CustomerID", sort=True)
    summary = grouped.agg(
        FirstPurchaseDate=("PurchaseDay", "min"),
        LastPurchaseDate=("PurchaseDay", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("LineAmount", "sum"),
        ProductDiversity=("StockCode", "nunique"),
    )
    summary["Recency"] = (reference - summary["LastPurchaseDate"]).dt.days
    summary["AvgOrderValue"] = summary["Monetary"] / summary["Frequency"]
    summary["ActiveDays"] = (summary["LastPurchaseDate"] - summary["FirstPurchaseDate"]).dt.days
    features = summary[FEATURE_COLUMNS].reset_index()
    dates = summary[["FirstPurchaseDate", "LastPurchaseDate"]].reset_index()
    return features, dates, reference
