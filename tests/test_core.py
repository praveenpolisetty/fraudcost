import numpy as np
from fraudcost import CostModel, calibrate, best_threshold, cost_recall_curve, expected_calibration_error


def _toy():
    rng = np.random.RandomState(0)
    n = 2000
    y = (rng.rand(n) < 0.05).astype(int)              # 5% fraud
    scores = np.clip(0.05 + 0.6 * y + rng.normal(0, 0.2, n), 0, 1)
    amounts = rng.gamma(2.0, 50.0, n)
    return y, scores, amounts


def test_costmodel_basic():
    y, s, a = _toy()
    cm = CostModel(admin_cost=5.0)
    r = cm.evaluate(y, s, a, 0.5)
    assert r["cost"] >= 0 and 0 <= r["recall"] <= 1 and r["false_positives"] >= 0


def test_best_threshold_reduces_cost():
    y, s, a = _toy()
    cm = CostModel(admin_cost=5.0)
    t = best_threshold(y, s, a, cm)
    assert cm.evaluate(y, s, a, t)["cost"] <= cm.evaluate(y, s, a, 0.5)["cost"]


def test_calibration_callable():
    y, s, a = _toy()
    for m in ("isotonic", "platt"):
        cal = calibrate(s, y, method=m)
        p = cal(s)
        assert p.shape == s.shape and p.min() >= 0 and p.max() <= 1


def test_curve_and_ece():
    y, s, a = _toy()
    cm = CostModel(admin_cost=5.0)
    curve = cost_recall_curve(y, s, a, cm)
    assert len(curve) > 0 and all("recall" in row for row in curve)
    assert expected_calibration_error(s, y) >= 0
