"""B's EDA on positive transactions and the six original customer features."""
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp/mpl_b"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing.feature_engineering import FEATURE_COLUMNS


def save_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def save_figure(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def describe_features(frame):
    values = frame[FEATURE_COLUMNS]
    table = values.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    table["skew"] = values.skew()
    table["missing"] = values.isna().sum()
    table["zeros"] = values.eq(0).sum()
    return table.rename_axis("Feature").reset_index()


def aggregate_orders(transactions):
    """An order is a (CustomerID, InvoiceNo) pair, never a detail row."""
    orders = transactions.groupby(["CustomerID", "InvoiceNo"], as_index=False).agg(
        OrderDate=("InvoiceDate", "min"), LastLineDate=("InvoiceDate", "max"),
        OrderAmount=("LineAmount", "sum"), DetailRows=("LineAmount", "size"),
    )
    if not orders.OrderDate.dt.to_period("M").equals(orders.LastLineDate.dt.to_period("M")):
        raise ValueError("An order spans months; resolve attribution before monthly analysis")
    return orders


def monthly_summary(transactions, orders, observation_start=None, observation_end=None):
    start = pd.Timestamp(observation_start if observation_start is not None else transactions.InvoiceDate.min()).normalize()
    end = pd.Timestamp(observation_end if observation_end is not None else transactions.InvoiceDate.max()).normalize()
    months = pd.period_range(start, end, freq="M")
    details = transactions.assign(Month=transactions.InvoiceDate.dt.to_period("M")).groupby("Month").agg(
        DetailRows=("InvoiceNo", "size"), SalesGBP=("LineAmount", "sum"))
    counts = orders.assign(Month=orders.OrderDate.dt.to_period("M")).groupby("Month").size()
    table = details.reindex(months, fill_value=0).rename_axis("Month")
    table["Orders"] = counts.reindex(months, fill_value=0)
    table["ObservedFrom"] = [max(start, month.start_time).date().isoformat() for month in months]
    table["ObservedThrough"] = [min(end, month.end_time.normalize()).date().isoformat() for month in months]
    table["CalendarDaysObserved"] = [(min(end, month.end_time.normalize()) - max(start, month.start_time)).days + 1 for month in months]
    table["CalendarDaysInMonth"] = months.days_in_month
    table["IsIncomplete"] = table.CalendarDaysObserved < table.CalendarDaysInMonth
    table["Month"] = months.astype(str)
    return table.reset_index(drop=True)


def concentration(customer_spend):
    descending = np.sort(np.asarray(customer_spend, dtype=float))[::-1]
    rows = []
    for share in [.01, .05, .10, .20]:
        count = math.ceil(len(descending) * share)
        rows.append({"TopCustomerFractionRequested": share, "Customers": count,
                     "ActualCustomerFraction": count / len(descending),
                     "SalesShare": descending[:count].sum() / descending.sum()})
    ascending = descending[::-1]
    lorenz = pd.DataFrame({"CustomerFraction": np.arange(len(ascending) + 1) / len(ascending),
                           "CumulativeSalesShare": np.r_[0, ascending.cumsum() / ascending.sum()]})
    gini = 2 * np.dot(np.arange(1, len(ascending) + 1), ascending) / (len(ascending) * ascending.sum()) - (len(ascending) + 1) / len(ascending)
    return pd.DataFrame(rows), lorenz, float(gini)


def run_eda(root, output, features):
    output = Path(output)
    output.mkdir(parents=True)
    transactions = pd.read_csv(root / "data/cleaned_data.csv",
        dtype={"CustomerID": "string", "InvoiceNo": "string", "StockCode": "string"},
        parse_dates=["InvoiceDate"])
    if transactions[["CustomerID", "InvoiceNo", "InvoiceDate", "LineAmount"]].isna().any().any():
        raise ValueError("Missing transaction keys, dates or amounts")
    if not np.isfinite(transactions.LineAmount).all() or transactions.LineAmount.le(0).any():
        raise ValueError("EDA requires finite positive transaction amounts")
    orders = aggregate_orders(transactions)
    grouped = orders.groupby("CustomerID").agg(Orders=("InvoiceNo", "size"), Monetary=("OrderAmount", "sum"))
    expected = features.set_index("CustomerID").reindex(grouped.index)
    if set(features.CustomerID) != set(grouped.index) or not np.array_equal(expected.Frequency, grouped.Orders):
        raise ValueError("Order counts disagree with customer features")
    np.testing.assert_allclose(expected.Monetary, grouped.Monetary, rtol=1e-12, atol=1e-8)
    manifest = json.loads((root / "data/dataset_manifest.json").read_text(encoding="utf-8"))
    monthly = monthly_summary(transactions, orders, manifest["purchase_date_min"], manifest["purchase_date_max"])
    top, lorenz, gini = concentration(grouped.Monetary)
    stats = describe_features(features)
    summary = {"detail_rows": len(transactions), "orders": len(orders), "customers": len(grouped),
               "positive_sales_gbp": float(orders.OrderAmount.sum()),
               "single_order_customers": int(grouped.Orders.eq(1).sum()),
               "single_order_fraction": float(grouped.Orders.eq(1).mean()), "gini": gini,
               "incomplete_months": monthly.loc[monthly.IsIncomplete, "Month"].tolist(),
               "order_amount_median": float(orders.OrderAmount.median()),
               "order_amount_p99": float(orders.OrderAmount.quantile(.99)),
               "order_amount_max": float(orders.OrderAmount.max())}
    for name, frame in [("orders", orders), ("monthly", monthly), ("spend_concentration", top),
                        ("lorenz", lorenz), ("feature_descriptive", stats),
                        ("customer_order_counts", grouped.reset_index()),
                        ("order_amount_descriptive", orders.OrderAmount.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).rename_axis("Statistic").reset_index(name="Value"))]:
        save_csv(frame, output / f"{name}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(orders.OrderAmount, bins=70)
    axes[0].set(xlabel="Order amount (GBP, positive purchases)", ylabel="Orders", title="Original scale: full range")
    axes[1].hist(np.log1p(orders.OrderAmount), bins=70)
    axes[1].set(xlabel="log1p(order amount)", ylabel="Orders", title="Log view of the same orders")
    save_figure(fig, output / "order_amount_distribution.png")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    colors = np.where(monthly.IsIncomplete, "#e79438", "#377eb8")
    for ax, column in zip(axes, ["Orders", "SalesGBP"]):
        ax.bar(monthly.Month, monthly[column], color=colors)
        ax.set(ylabel=column, title="Orange = incomplete observation month")
    axes[-1].tick_params(axis="x", rotation=45)
    save_figure(fig, output / "monthly_orders_sales.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(["One order", "Repeat orders"], [summary["single_order_customers"], len(grouped) - summary["single_order_customers"]])
    axes[0].set(ylabel="Customers", title=f"One-order share: {summary['single_order_fraction']:.1%}")
    axes[1].plot(lorenz.CustomerFraction, lorenz.CumulativeSalesShare, label=f"Gini = {gini:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].set(xlabel="Customers, ascending spend", ylabel="Cumulative positive sales share")
    axes[1].legend()
    save_figure(fig, output / "single_order_and_concentration.png")
    for log_view in [False, True]:
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        for ax, column in zip(axes.flat, FEATURE_COLUMNS):
            values = np.log1p(features[column]) if log_view else features[column]
            ax.hist(values, bins=55)
            ax.set(xlabel=f"log1p({column})" if log_view else column, ylabel="Customers")
        fig.suptitle("Visualization only: log1p of all six features" if log_view else "All six features: original scale, full range")
        save_figure(fig, output / ("feature_distributions_log_view.png" if log_view else "feature_distributions_raw.png"))
    correlations = {}
    for method in ["pearson", "spearman"]:
        matrix = features[FEATURE_COLUMNS].corr(method=method)
        correlations[method] = matrix
        save_csv(matrix.rename_axis("Feature").reset_index(), output / f"correlation_{method}.csv")
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set(xticks=range(6), yticks=range(6), xticklabels=FEATURE_COLUMNS, yticklabels=FEATURE_COLUMNS, title=f"{method.title()}: original six features")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", color="white" if abs(matrix.iloc[i, j]) > .65 else "black")
        fig.colorbar(im, ax=ax)
        save_figure(fig, output / f"correlation_{method}.png")
    pairs = pd.DataFrame([{"Feature1": a, "Feature2": b,
                           **{method: float(matrix.loc[a, b]) for method, matrix in correlations.items()}}
                          for i, a in enumerate(FEATURE_COLUMNS) for b in FEATURE_COLUMNS[i + 1:]])
    pairs["MaxAbsoluteCorrelation"] = pairs[["pearson", "spearman"]].abs().max(axis=1)
    pairs = pairs.sort_values("MaxAbsoluteCorrelation", ascending=False)
    save_csv(pairs, output / "correlation_pairs.csv")
    diagnostics = pd.read_csv(root / "results/tables/customer_cancellation_diagnostics.csv", dtype={"CustomerID": "string"})
    candidates = pd.read_csv(root / "results/tables/largest_order_cancellation_candidates.csv", dtype={"CustomerID": "string", "InvoiceNo": "string", "InvoiceNo_Cancellation": "string"})
    extreme_ids = pd.concat([features.nlargest(20, "Monetary"), features.nlargest(20, "AvgOrderValue")]).CustomerID.drop_duplicates()
    extreme = features[features.CustomerID.isin(extreme_ids)].merge(diagnostics.drop(columns="Monetary"), on="CustomerID", validate="one_to_one")
    extreme["CancellationAmountToPositiveSales"] = extreme.CancellationAmountAbs / extreme.Monetary
    extreme["HasLaterCancellationCandidate"] = extreme.CustomerID.isin(candidates.loc[candidates.InvoiceNo_Cancellation.notna(), "CustomerID"])
    save_csv(extreme.sort_values("Monetary", ascending=False), output / "extreme_customer_cancellation_audit.csv")
    save_csv(candidates, output / "existing_cancellation_candidates.csv")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary, stats, pairs, top
