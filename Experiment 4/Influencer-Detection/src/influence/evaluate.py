"""Out-of-fold evaluation: every number reported comes from nodes the model was not fitted on."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def transform(X: np.ndarray) -> np.ndarray:
    """Signed log: tames heavy tails (degree, triangles, CI) and keeps the sign of curvature."""
    return np.sign(X) * np.log1p(np.abs(X))


def oof(X: np.ndarray, y_rank: np.ndarray, label: np.ndarray, folds: int = 5, seed: int = 0) -> dict:
    """Fit a classifier (label) and a regressor (y_rank = log spread) per fold; score on the held-out fold."""
    Z = transform(X)
    p = np.zeros(len(Z)); r = np.zeros(len(Z))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(Z, label):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(Z[tr], label[tr])
        p[te] = clf.predict_proba(Z[te])[:, 1]
        reg = make_pipeline(StandardScaler(), LinearRegression()).fit(Z[tr], y_rank[tr])
        r[te] = reg.predict(Z[te])
    return {"roc_auc": roc_auc_score(label, p), "avg_precision": average_precision_score(label, p),
            "r2": r2_score(y_rank, r), "spearman": spearmanr(r, y_rank)[0]}


def unfitted(score: np.ndarray, y_rank: np.ndarray, label: np.ndarray) -> dict:
    """A single feature used directly as the score; R2 from a one-variable linear fit, out of fold."""
    out = oof(score.reshape(-1, 1), y_rank, label)
    out["roc_auc"] = roc_auc_score(label, score)                       # no fitting needed for a ranking
    out["avg_precision"] = average_precision_score(label, score)
    return out
