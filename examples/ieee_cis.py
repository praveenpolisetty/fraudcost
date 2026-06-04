"""Reproduce the paper's IEEE-CIS result using the fraudcost library.

This is a thin wrapper: train any model, then let fraudcost handle calibration,
cost-optimal thresholding, and the cost-recall curve. Mirrors the standalone
experiments.py shipped with the paper, but using the public API.

Usage:
  python examples/ieee_cis.py --data_dir /path/to/ieee-cis --ca 5
"""
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from fraudcost import CostModel, calibrate, best_threshold, cost_recall_curve, expected_calibration_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ca", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tx = pd.read_csv(f"{args.data_dir}/train_transaction.csv")
    try:
        df = tx.merge(pd.read_csv(f"{args.data_dir}/train_identity.csv"), on="TransactionID", how="left")
    except FileNotFoundError:
        df = tx
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df); a = int(n*0.6); b = int(n*0.8)
    tr, ca, te = df.iloc[:a], df.iloc[a:b], df.iloc[b:]

    drop = {"isFraud", "TransactionID", "TransactionDT"}
    feat = [c for c in tr.columns if c not in drop]
    cat = [c for c in feat if tr[c].dtype == "object"]
    freq = {c: tr[c].value_counts(dropna=False) for c in cat}
    def prep(f):
        f = f[feat].copy()
        for c in cat: f[c] = f[c].map(freq[c]).fillna(0)
        return f.apply(pd.to_numeric, errors="coerce").fillna(-999)
    Xtr, Xca, Xte = prep(tr), prep(ca), prep(te)

    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.02, num_leaves=256,
                           subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                           random_state=args.seed, n_jobs=-1)
    m.fit(Xtr, tr.isFraud.values)
    s_ca, s_te = m.predict_proba(Xca)[:, 1], m.predict_proba(Xte)[:, 1]

    cal = calibrate(s_ca, ca.isFraud.values, method="isotonic")
    cm = CostModel(admin_cost=args.ca)
    t = best_threshold(ca.isFraud.values, cal(s_ca), ca.TransactionAmt.values, cm)

    base = cm.evaluate(te.isFraud.values, s_te, te.TransactionAmt.values, 0.5)
    opt = cm.evaluate(te.isFraud.values, cal(s_te), te.TransactionAmt.values, t)
    print("ECE raw/cal:", round(expected_calibration_error(s_te, te.isFraud.values), 4),
          round(expected_calibration_error(cal(s_te), te.isFraud.values), 4))
    print("baseline (raw, t=0.5):", base)
    print("calibrated, cost-opt :", opt)
    print(f"cost saved: ${base['cost'] - opt['cost']:,.0f}")
    pd.DataFrame(cost_recall_curve(te.isFraud.values, cal(s_te), te.TransactionAmt.values, cm)) \
        .to_csv("cost_recall_curve.csv", index=False)


if __name__ == "__main__":
    main()
