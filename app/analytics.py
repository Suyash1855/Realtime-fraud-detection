"""
Analytics over precomputed demo scores
======================================
Every transaction in the demo set is scored once at startup. Everything here is
then pure arithmetic over those scores, which is what makes the threshold slider
feel instant -- moving it re-derives precision, recall, cost and the confusion
matrix without touching the model.

All figures are genuine out-of-sample: the demo rows come from the chronological
test period (see scripts/build_demo_data.py).
"""

from __future__ import annotations

import numpy as np


class Analytics:
    def __init__(self, scores: np.ndarray, labels: np.ndarray, amounts: np.ndarray):
        self.scores = np.asarray(scores, dtype=np.float64)
        self.labels = np.asarray(labels, dtype=np.int8)
        self.amounts = np.nan_to_num(np.asarray(amounts, dtype=np.float64))

        self.n = len(self.scores)
        self.n_fraud = int(self.labels.sum())
        self.n_legit = self.n - self.n_fraud
        self.fraud_amount_total = float(self.amounts[self.labels == 1].sum())

        # Sort once; every threshold query is then a binary search plus a
        # cumulative-sum lookup rather than a full pass over the data.
        order = np.argsort(-self.scores, kind="stable")
        self._sorted_scores = self.scores[order]
        self._sorted_labels = self.labels[order]
        self._cum_tp = np.cumsum(self._sorted_labels)
        self._cum_amount_caught = np.cumsum(self.amounts[order] * self._sorted_labels)

    # ── threshold evaluation ─────────────────────────────────────────────────

    def at_threshold(self, threshold: float) -> dict:
        """
        Full confusion matrix and business impact at one operating point.
        O(log n) rather than O(n), so this is safe to call on every slider tick.
        """
        # Number of rows scoring >= threshold, via the descending-sorted array.
        k = int(np.searchsorted(-self._sorted_scores, -threshold, side="right"))

        tp = int(self._cum_tp[k - 1]) if k > 0 else 0
        amount_caught = float(self._cum_amount_caught[k - 1]) if k > 0 else 0.0
        fp = k - tp
        fn = self.n_fraud - tp
        tn = self.n_legit - fp

        precision = tp / k if k else 0.0
        recall = tp / self.n_fraud if self.n_fraud else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        fpr = fp / self.n_legit if self.n_legit else 0.0

        return {
            "threshold": round(float(threshold), 4),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "alerts": k,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "alert_rate": round(k / self.n, 4) if self.n else 0.0,
            "fraud_amount_caught": round(amount_caught, 2),
            "fraud_amount_missed": round(self.fraud_amount_total - amount_caught, 2),
            "fraud_amount_total": round(self.fraud_amount_total, 2),
            "capture_rate": (
                round(amount_caught / self.fraud_amount_total, 4)
                if self.fraud_amount_total else 0.0
            ),
        }

    # ── curves ───────────────────────────────────────────────────────────────

    def pr_curve(self, n_points: int = 120) -> list:
        """Precision/recall across the threshold range, downsampled for plotting."""
        thresholds = np.unique(
            np.quantile(self.scores, np.linspace(0.0, 1.0, n_points))
        )
        out = []
        for t in thresholds:
            m = self.at_threshold(float(t))
            if m["alerts"] == 0:
                continue
            out.append({
                "threshold": m["threshold"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "alerts": m["alerts"],
                "false_positives": m["false_positives"],
            })
        return out

    def roc_curve(self, n_points: int = 120) -> list:
        thresholds = np.unique(
            np.quantile(self.scores, np.linspace(0.0, 1.0, n_points))
        )
        out = []
        for t in thresholds:
            m = self.at_threshold(float(t))
            out.append({
                "threshold": m["threshold"],
                "tpr": m["recall"],
                "fpr": m["false_positive_rate"],
            })
        return out

    def score_distribution(self, bins: int = 40) -> dict:
        """
        Score histogram split by true label. This is the chart that makes model
        quality legible at a glance: good separation means two distinct humps.
        """
        edges = np.linspace(0.0, 1.0, bins + 1)
        fraud_hist, _ = np.histogram(self.scores[self.labels == 1], bins=edges)
        legit_hist, _ = np.histogram(self.scores[self.labels == 0], bins=edges)
        return {
            "bin_edges": [round(float(e), 4) for e in edges],
            "fraud": [int(v) for v in fraud_hist],
            "legitimate": [int(v) for v in legit_hist],
        }

    # ── cost model ───────────────────────────────────────────────────────────

    def cost_sweep(self, fp_cost: float, fn_cost: float, n_points: int = 120) -> dict:
        """
        The question an interviewer actually cares about: what does this threshold
        cost the business?

        fp_cost -- cost of blocking/reviewing one legitimate transaction
        fn_cost -- cost of letting one fraudulent transaction through

        Returns the sweep plus the cost-minimising threshold.
        """
        thresholds = np.unique(
            np.quantile(self.scores, np.linspace(0.0, 1.0, n_points))
        )
        points, best, best_cost = [], None, float("inf")

        for t in thresholds:
            m = self.at_threshold(float(t))
            total = m["false_positives"] * fp_cost + m["false_negatives"] * fn_cost
            points.append({
                "threshold": m["threshold"],
                "total_cost": round(total, 2),
                "fp_cost": round(m["false_positives"] * fp_cost, 2),
                "fn_cost": round(m["false_negatives"] * fn_cost, 2),
                "precision": m["precision"],
                "recall": m["recall"],
            })
            if total < best_cost:
                best_cost, best = total, m["threshold"]

        return {
            "points": points,
            "optimal_threshold": best,
            "optimal_cost": round(best_cost, 2),
            "fp_cost": fp_cost,
            "fn_cost": fn_cost,
        }

    def summary(self) -> dict:
        return {
            "n_transactions": self.n,
            "n_fraud": self.n_fraud,
            "fraud_rate": round(self.n_fraud / self.n, 4) if self.n else 0.0,
            "fraud_amount_total": round(self.fraud_amount_total, 2),
            "amount_total": round(float(self.amounts.sum()), 2),
        }
