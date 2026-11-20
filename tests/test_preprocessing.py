"""Small hand-calculated examples covering the actual data contract."""
import unittest

import numpy as np
import pandas as pd

from preprocessing.data_cleaning import clean_transactions
from preprocessing.feature_engineering import build_customer_features
from preprocessing.diagnostics import return_diagnostics


def record(**changes):
    value = {"InvoiceNo": "100001", "StockCode": "A1", "Description": "item",
             "Quantity": 2, "InvoiceDate": "2011-01-01 23:59:00",
             "UnitPrice": 10.0, "CustomerID": "12345", "Country": "United Kingdom"}
    value.update(changes)
    return value


class PreprocessingTests(unittest.TestCase):
    def test_six_features_from_multiple_lines_and_orders(self):
        raw = pd.DataFrame([
            record(), record(StockCode="B1", Quantity=1, UnitPrice=5),
            record(InvoiceNo="100002", InvoiceDate="2011-01-03 00:01:00", Quantity=3),
            record(InvoiceNo="100003", CustomerID="00007", InvoiceDate="2011-01-03", Quantity=1),
        ])
        clean, _, _ = clean_transactions(raw)
        features, _, date = build_customer_features(clean)
        first = features.set_index("CustomerID").loc["12345"]
        self.assertEqual(date, pd.Timestamp("2011-01-04"))
        self.assertEqual(first.to_dict(), {"Recency": 1, "Frequency": 2, "Monetary": 55,
                                          "AvgOrderValue": 27.5, "ProductDiversity": 2, "ActiveDays": 2})
        self.assertIn("00007", features["CustomerID"].tolist())
        self.assertEqual(features.set_index("CustomerID").loc["00007", "ActiveDays"], 0)

    def test_filter_reasons_are_exclusive(self):
        raw = pd.DataFrame([
            record(), record(CustomerID=None, InvoiceNo="C100002", Quantity=-2),
            record(InvoiceNo="C100003", Quantity=-2), record(InvoiceNo="100004", Quantity=0),
            record(InvoiceNo="100005", UnitPrice=0), record(InvoiceNo="100006", UnitPrice=np.inf),
            record(InvoiceNo="100007", InvoiceDate="invalid"), record(InvoiceNo="100008", StockCode=None),
        ])
        clean, excluded, ledger = clean_transactions(raw)
        self.assertEqual(len(clean), 1)
        self.assertEqual(len(excluded), 7)
        self.assertEqual(ledger.removed_rows.sum(), 7)
        self.assertEqual(excluded.loc[excluded.SourceRow.eq(3), "ExclusionReason"].item(), "missing_customer_id")
        self.assertIn("cancelled_invoice", excluded.ExclusionReason.tolist())

    def test_exact_duplicate_policy_is_reversible(self):
        raw = pd.DataFrame([record(), record(), record(Quantity=3)])
        dropped, rejected, _ = clean_transactions(raw)
        kept, _, _ = clean_transactions(raw, "keep")
        self.assertEqual((len(dropped), len(kept)), (2, 3))
        self.assertEqual(rejected.ExclusionReason.item(), "exact_duplicate")
        self.assertEqual(dropped.SourceRow.tolist(), [2, 4])

    def test_extreme_and_special_codes_are_preserved(self):
        raw = pd.DataFrame([record(StockCode="POST", Quantity=80000, UnitPrice=99999)])
        clean, rejected, _ = clean_transactions(raw)
        self.assertEqual(len(clean), 1)
        self.assertTrue(rejected.empty)

    def test_decimal_customer_id_and_whitespace(self):
        clean, _, _ = clean_transactions(pd.DataFrame([record(CustomerID=" 12345.0 ", InvoiceNo=" 100001 ")]))
        self.assertEqual(clean.CustomerID.item(), "12345")
        self.assertEqual(clean.InvoiceNo.item(), "100001")

    def test_invalid_reference_date_is_rejected(self):
        clean, _, _ = clean_transactions(pd.DataFrame([record()]))
        with self.assertRaises(ValueError):
            build_customer_features(clean, "2011-01-01")

    def test_empty_cleaned_input_cannot_produce_delivery(self):
        clean, _, _ = clean_transactions(pd.DataFrame([record(Quantity=-1)]))
        with self.assertRaises(ValueError):
            build_customer_features(clean)

    def test_subpenny_price_not_rounded_before_aggregation(self):
        clean, _, _ = clean_transactions(pd.DataFrame([record(Quantity=3, UnitPrice=.001)]))
        features, _, _ = build_customer_features(clean)
        self.assertAlmostEqual(features.Monetary.item(), .003)

    def test_return_diagnostics_do_not_net_features(self):
        raw = pd.DataFrame([record(), record(InvoiceNo="C100001", Quantity=-2,
                                          InvoiceDate="2011-01-02")])
        clean, _, _ = clean_transactions(raw)
        features, _, _ = build_customer_features(clean)
        customers, pairs = return_diagnostics(raw, clean, features, "drop_exact")
        self.assertEqual(features.Monetary.item(), 20)
        self.assertEqual(customers.CancellationAmountAbs.item(), 20)
        self.assertEqual(customers.CancelledInvoices.item(), 1)
        self.assertEqual(len(pairs), 1)


if __name__ == "__main__":
    unittest.main()
