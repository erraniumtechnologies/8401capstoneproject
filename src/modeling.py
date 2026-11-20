"""Splitting, preprocessing, model factories, metrics, thresholds and calibration."""
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config as C

# Harmless: LightGBM vs scikit-learn feature-name bookkeeping (column order is identical by construction)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

try:  # LightGBM is the production backend; sklearn's HistGradientBoosting is a documented fallback
    import lightgbm as lgbm
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:  # pragma: no cover
    HAS_LGBM = False
    from sklearn.ensemble import HistGradientBoostingClassifier


def backend() -> str:
    return "lightgbm" if HAS_LGBM else "sklearn-histgb (fallback)"


def make_preprocessor(nums=None, cats=None) -> ColumnTransformer:
    nums = C.NUMS if nums is None else nums
    cats = C.CATS if cats is None else cats
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), nums),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), cats)])


def grouped_split(d: pd.DataFrame, test_size: float, seed: int):
    a, b = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed)
                .split(d, d[C.TARGET], d[C.CLIENT]))
    return d.iloc[a], d.iloc[b]


def three_way_split(df):
    train_full, test = grouped_split(df, 0.25, C.RANDOM)
    train, val = grouped_split(train_full, 0.20, 1)
    assert not (set(train_full[C.CLIENT]) & set(test[C.CLIENT])), "client leakage across train/test"
    assert not (set(train[C.CLIENT]) & set(val[C.CLIENT])), "client leakage across train/val"
    return train, val, test


def fit_gbm(Xtr, ytr, Xva, yva, params, n_estimators=3000, early_stopping=True):
    """Fit the tuned gradient-boosted model. Returns (model, n_trees)."""
    if HAS_LGBM:
        m = LGBMClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=C.RANDOM,
                           verbose=-1, **params)
        if early_stopping:
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
                  callbacks=[lgbm.early_stopping(50, verbose=False)])
            return m, (m.best_iteration_ or n_estimators)
        m.fit(Xtr, ytr)
        return m, n_estimators
    # fallback: map the LightGBM search space onto HistGradientBoosting
    m = HistGradientBoostingClassifier(
        learning_rate=max(params["learning_rate"], 0.05), max_leaf_nodes=params["num_leaves"],
        min_samples_leaf=params["min_child_samples"], l2_regularization=params["reg_lambda"],
        class_weight={0: 1.0, 1: params["scale_pos_weight"]}, max_iter=min(n_estimators, 300),
        early_stopping=early_stopping, validation_fraction=0.1, n_iter_no_change=20,
        random_state=C.RANDOM)
    m.fit(Xtr, ytr)
    return m, getattr(m, "n_iter_", 300)


def ks_stat(y, p):
    f, t, _ = roc_curve(y, p)
    return float(np.max(t - f))


def thresholds_from_validation(y_val, p_val, prec_floor=C.PREC_FLOOR,
                               value_tp=C.VALUE_TP, cost_fp=C.COST_FP):
    y_val = np.asarray(y_val)
    pr, rc, th = precision_recall_curve(y_val, p_val)
    f1v = 2 * pr * rc / (pr + rc + 1e-9)
    t_f1 = float(th[int(np.argmax(f1v[:-1]))])
    idx = np.where(pr[:-1] >= prec_floor)[0]
    t_floor = float(th[idx[0]]) if len(idx) else 1.0
    grid = np.linspace(0.02, 0.98, 193)
    profit = [value_tp * ((p_val >= t) & (y_val == 1)).sum() - cost_fp * ((p_val >= t) & (y_val == 0)).sum()
              for t in grid]
    return dict(f1=t_f1, recall_at_floor=t_floor, profit=float(grid[int(np.argmax(profit))])), grid, profit


def full_metrics(y_true, p_test, thr_f1, thr_floor):
    y = np.asarray(y_true)
    order = np.argsort(p_test)[::-1]

    def topk(pct):
        k = max(1, int(len(p_test) * pct))
        return float(y[order[:k]].mean())

    yh = (p_test >= thr_f1).astype(int)
    yf = (p_test >= thr_floor).astype(int)
    return dict(
        ROC_AUC=roc_auc_score(y, p_test), PR_AUC=average_precision_score(y, p_test), KS=ks_stat(y, p_test),
        Prec_top5=topk(0.05), Prec_top10=topk(0.10), Prec_top20=topk(0.20), Lift_top10=topk(0.10) / y.mean(),
        Precision=precision_score(y, yh, zero_division=0), Recall=recall_score(y, yh, zero_division=0),
        F1=f1_score(y, yh, zero_division=0),
        P_at_floor=precision_score(y, yf, zero_division=0), R_at_floor=recall_score(y, yf, zero_division=0))


def fit_calibrators(p_val, y_val, p_test, y_test):
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_val, y_val)
    platt = LogisticRegression(max_iter=1000).fit(p_val.reshape(-1, 1), y_val)
    cands = {"raw": p_test, "isotonic": iso.transform(p_test),
             "platt": platt.predict_proba(p_test.reshape(-1, 1))[:, 1]}
    briers = {k: float(brier_score_loss(y_test, v)) for k, v in cands.items()}
    best = min(briers, key=briers.get)
    return {"raw": None, "isotonic": iso, "platt": platt}[best], best, briers


def apply_calibrator(cal, cal_type, p):
    p = np.asarray(p, dtype=float)
    if cal is None:
        return p
    if cal_type == "isotonic":
        return cal.transform(p)
    return cal.predict_proba(p.reshape(-1, 1))[:, 1]
