#!/usr/bin/env python3
"""How stable are the headline numbers? Different cross-validation splits, and a bootstrap interval."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from influence.evaluate import oof                                     # noqa: E402
from influence.features import FEATURES                                # noqa: E402
from sklearn.metrics import roc_auc_score

t = pd.read_csv("results/tables/seed_table.csv")
y, label = np.log(t["spread"].to_numpy()), t["influencer"].to_numpy()
dct, allf = t[["degree", "coreness", "trussness"]].to_numpy(), t[FEATURES].to_numpy()
rows = []
for name, X in (("D-C-T ensemble", dct), ("all 20 features", allf)):
    runs = [oof(X, y, label, seed=s) for s in range(10)]
    a, r = np.array([x["roc_auc"] for x in runs]), np.array([x["r2"] for x in runs])
    rows.append({"model": name, "roc_auc_mean": a.mean(), "roc_auc_min": a.min(), "roc_auc_max": a.max(), "r2_mean": r.mean(), "r2_min": r.min(), "r2_max": r.max()})
print(pd.DataFrame(rows).round(4).to_string(index=False))
deg = t["degree"].to_numpy(); rng = np.random.default_rng(0); boots = []
for _ in range(2000):
    i = rng.integers(0, len(t), len(t))
    if label[i].min() != label[i].max(): boots.append(roc_auc_score(label[i], deg[i]))
print(f"degree-alone ROC-AUC 95% bootstrap interval: {np.percentile(boots, 2.5):.4f} to {np.percentile(boots, 97.5):.4f}")
print(f"positives: {label.sum()} of {len(label)}; spread threshold for 'influencer': {t['spread'][label == 1].min():.1f} nodes")
