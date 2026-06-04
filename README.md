# fraudcost — cost-aware thresholding & calibration for fraud models

[![CI](https://github.com/praveenpolisetty/fraudcost/actions/workflows/ci.yml/badge.svg)](https://github.com/praveenpolisetty/fraudcost/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#install)

**Your fraud model is probably pointed at the wrong number.** Most detectors are tuned for AUC, but a
deployed system decides at a threshold where the two kinds of error cost very different amounts.
`fraudcost` is a tiny, dependency-light library that takes any classifier's scores and gives you:

- **Example-dependent cost** scoring (false negative = transaction amount, false positive = admin cost),
- **Probability calibration** (Platt / isotonic) so scores behave like probabilities,
- **Cost-optimal threshold** selection, and
- The **cost-vs-recall operating curve** that makes the trade-off legible.

No new model. It wraps the scores you already have.

> Companion research: *Paying for Precision: A Cost-Aware Evaluation of Card-Fraud Models on the
> IEEE-CIS Benchmark* (paper in preparation). Reproduces the paper's numbers via `examples/ieee_cis.py`.

## Install
```bash
pip install fraudcost          # once published
# or, from source:
pip install -e .
```

## Quickstart
```python
import numpy as np
from fraudcost import CostModel, best_threshold, calibrate, cost_recall_curve

# y: 0/1 labels, scores: model probabilities, amounts: transaction amounts
cm = CostModel(admin_cost=5.0)                      # $5 per review (false positive / flagged)
cal = calibrate(scores_calib, y_calib, method="isotonic")
scores_cal = cal(scores_test)

t = best_threshold(y_calib, cal(scores_calib), amounts_calib, cm)   # pick on calib slice
report = cm.evaluate(y_test, scores_cal, amounts_test, t)           # apply to test
print(report)        # {'cost': ..., 'false_positives': ..., 'recall': ..., 'threshold': t}

curve = cost_recall_curve(y_test, scores_cal, amounts_test, cm)     # for plotting
```

## Why it exists
Calibration and cost-sensitive thresholding are two decades old (Elkan 2001; Zadrozny & Elkan 2002;
Bahnsen et al. 2015), yet they rarely make it into how fraud models are *evaluated and reported*, where
AUC still rules. `fraudcost` makes the cost-aware lens a one-import step so teams can recalibrate and
rethreshold often — cheap, fast, and far less risky than retraining.

## API
| Function | Purpose |
|----------|---------|
| `CostModel(admin_cost, fn_cost="amount")` | defines the example-dependent cost matrix |
| `calibrate(scores, y, method)` | returns a calibration map (`"platt"` or `"isotonic"`) |
| `best_threshold(y, scores, amounts, cost_model)` | cost-minimizing threshold |
| `CostModel.evaluate(y, scores, amounts, t)` | cost / FP / recall at a threshold |
| `cost_recall_curve(y, scores, amounts, cost_model)` | DataFrame for the operating curve |
| `expected_calibration_error(p, y)` | ECE for reliability checks |

## Reproduce the paper
```bash
python examples/ieee_cis.py --data_dir /path/to/ieee-cis --ca 5
```

## Roadmap
- [ ] AML / graph support (Elliptic dataset example)
- [ ] sklearn-compatible `CostAwareClassifier` wrapper
- [ ] cost-curve plotting helpers
- [ ] PyPI release

## Citing
If you use `fraudcost`, please cite the companion paper (see `CITATION.cff`).

## Contributing
Issues and PRs welcome — see `CONTRIBUTING.md`. Licensed under MIT.
