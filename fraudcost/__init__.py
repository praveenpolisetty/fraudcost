"""fraudcost — cost-aware thresholding & calibration for fraud models.

A tiny, dependency-light toolkit that wraps any classifier's scores with
example-dependent cost evaluation, probability calibration, and cost-optimal
thresholding. See README for the rationale and the companion paper.
"""
from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

__version__ = "0.1.0"
__all__ = ["CostModel", "calibrate", "best_threshold", "cost_recall_curve",
           "expected_calibration_error"]


class CostModel:
    """Example-dependent cost model for fraud decisions.

    false negative  -> costs the transaction amount (fraud allowed through)
    false positive  -> costs ``admin_cost`` (legit transaction reviewed/blocked)
    true positive   -> costs ``admin_cost`` (flagged fraud still consumes a review)
    true negative   -> 0
    """

    def __init__(self, admin_cost: float, fn_cost: str = "amount"):
        if admin_cost < 0:
            raise ValueError("admin_cost must be non-negative")
        self.admin_cost = float(admin_cost)
        self.fn_cost = fn_cost

    def evaluate(self, y, scores, amounts, threshold: float) -> dict:
        y = np.asarray(y); scores = np.asarray(scores); amounts = np.asarray(amounts)
        pred = (scores >= threshold).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        tp = int(((pred == 1) & (y == 1)).sum())
        fn_mask = (pred == 0) & (y == 1)
        cost = (fp + tp) * self.admin_cost + float(amounts[fn_mask].sum())
        recall = tp / max(int(y.sum()), 1)
        return {"cost": float(cost), "false_positives": fp, "true_positives": tp,
                "recall": float(recall), "threshold": float(threshold)}


def calibrate(scores, y, method: str = "isotonic"):
    """Fit a calibration map on (scores, y); return a callable map: scores -> probs."""
    scores = np.asarray(scores, dtype=float); y = np.asarray(y)
    if method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip"); ir.fit(scores, y)
        return lambda s: ir.predict(np.asarray(s, dtype=float))
    if method == "platt":
        lr = LogisticRegression(max_iter=1000); lr.fit(scores.reshape(-1, 1), y)
        return lambda s: lr.predict_proba(np.asarray(s, dtype=float).reshape(-1, 1))[:, 1]
    raise ValueError("method must be 'isotonic' or 'platt'")


def best_threshold(y, scores, amounts, cost_model: CostModel, grid: int = 200) -> float:
    """Threshold minimizing expected cost, searched over score quantiles."""
    scores = np.asarray(scores, dtype=float)
    ts = np.quantile(scores, np.linspace(0.5, 0.9999, grid))
    costs = [cost_model.evaluate(y, scores, amounts, t)["cost"] for t in ts]
    return float(ts[int(np.argmin(costs))])


def cost_recall_curve(y, scores, amounts, cost_model: CostModel, points: int = 100):
    """Return a list of {cost, recall, threshold, ...} dicts swept over the threshold."""
    scores = np.asarray(scores, dtype=float)
    ts = np.quantile(scores, np.linspace(0.5, 0.9999, points))
    rows = []
    for t in ts:
        r = cost_model.evaluate(y, scores, amounts, t); rows.append(r)
    return rows


def expected_calibration_error(p, y, bins: int = 15) -> float:
    p = np.asarray(p, dtype=float); y = np.asarray(y)
    edges = np.linspace(0, 1, bins + 1); ece = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(ece)
