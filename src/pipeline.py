"""End-to-end, scripted pipeline: ingest -> features -> split -> tune -> train -> evaluate -> save.

Usage:
    python -m src.pipeline                      # full run (Optuna tuning, CV, OOT)
    python -m src.pipeline --no-tune            # reuse the tuned hyper-parameters from config.py
    python -m src.pipeline --sample-clients 5000 --no-tune --skip-cv   # fast smoke test
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import features as F
from . import modeling as M


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tune(Xtr, ytr, Xva, yva, n_trials):
    """Bayesian (TPE) search with validation PR-AUC as the objective."""
    try:
        import optuna
    except ImportError:
        log("optuna not installed: using configured best parameters")
        return dict(C.BEST_PARAMS)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        p = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 160),
            min_child_samples=trial.suggest_int("min_child_samples", 30, 300),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 3.0),
        )
        m, _ = M.fit_gbm(Xtr, ytr, Xva, yva, p)
        return average_precision_score(yva, m.predict_proba(Xva)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.RANDOM))
    study.optimize(objective, n_trials=n_trials)
    log(f"Optuna best validation PR-AUC {study.best_value:.4f}")
    return study.best_params


def grouped_repeated_cv(Xtr, ytr, gtr, params, n_trees, n_splits=4, n_repeats=2):
    rng = np.random.default_rng(C.RANDOM)
    uniq = gtr.unique()
    roc, prc = [], []
    for _ in range(n_repeats):
        folds = np.array_split(rng.permutation(uniq), n_splits)
        for f in folds:
            mask = gtr.isin(set(f)).values
            m, _ = M.fit_gbm(Xtr[~mask], ytr[~mask], None, None, params, n_estimators=n_trees,
                             early_stopping=False)
            p = m.predict_proba(Xtr[mask])[:, 1]
            roc.append(roc_auc_score(ytr[mask], p))
            prc.append(average_precision_score(ytr[mask], p))

    def ci(a):
        a = np.array(a)
        half = 1.96 * a.std() / np.sqrt(len(a))
        return [float(a.mean()), float(a.mean() - half), float(a.mean() + half)]
    return ci(roc), ci(prc)


def out_of_time(df, prep, params, n_trees):
    cut = df["Loan_Open_Date"].quantile(0.70)
    a, b = df[df["Loan_Open_Date"] <= cut], df[df["Loan_Open_Date"] > cut]
    pp = clone(prep)
    A, B = pp.fit_transform(a[C.FEATURES]), pp.transform(b[C.FEATURES])
    m, _ = M.fit_gbm(A, a[C.TARGET], None, None, params, n_estimators=n_trees, early_stopping=False)
    p = m.predict_proba(B)[:, 1]
    return dict(cut=str(cut.date()), n_train=int(len(a)), n_test=int(len(b)),
                ROC_AUC=float(roc_auc_score(b[C.TARGET], p)), PR_AUC=float(average_precision_score(b[C.TARGET], p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--sample-clients", type=int, default=0, help="subsample borrowers (smoke tests)")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--trials", type=int, default=C.N_TRIALS)
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    C.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"backend: {M.backend()}")

    # 1. ingest + target
    df = F.add_target(F.load_raw(args.data))
    if args.sample_clients:
        keep = df[C.CLIENT].drop_duplicates().sample(args.sample_clients, random_state=C.RANDOM)
        df = df[df[C.CLIENT].isin(set(keep))]
    log(f"loans {len(df):,} | clients {df[C.CLIENT].nunique():,} | target rate {df[C.TARGET].mean():.4f}")

    # 2. features
    df = F.build_features(df)
    df = df.sort_values("Loan_Open_Date").reset_index(drop=True)

    # 3. grouped split + preprocessing (fit on train only)
    train, val, test = M.three_way_split(df)
    prep = M.make_preprocessor()
    Xtr, Xva, Xte = (prep.fit_transform(train[C.FEATURES]), prep.transform(val[C.FEATURES]),
                     prep.transform(test[C.FEATURES]))
    ytr, yva, yte = train[C.TARGET].values, val[C.TARGET].values, test[C.TARGET].values
    log(f"train {len(train):,} | val {len(val):,} | test {len(test):,} | design matrix {Xtr.shape[1]} cols")

    # 4. tuning + champion
    params = dict(C.BEST_PARAMS) if args.no_tune else tune(Xtr, ytr, Xva, yva, args.trials)
    gbm, n_trees = M.fit_gbm(Xtr, ytr, Xva, yva, params)
    val_p = {"GBM_tuned": gbm.predict_proba(Xva)[:, 1]}
    test_p = {"GBM_tuned": gbm.predict_proba(Xte)[:, 1]}

    # 5. baselines
    if not args.skip_baselines:
        spw = params["scale_pos_weight"]
        sc = StandardScaler(with_mean=False)
        lr = LogisticRegression(max_iter=3000, C=0.5, class_weight={0: 1, 1: spw}).fit(sc.fit_transform(Xtr), ytr)
        val_p["LogReg"], test_p["LogReg"] = (lr.predict_proba(sc.transform(Xva))[:, 1],
                                             lr.predict_proba(sc.transform(Xte))[:, 1])
        rf = RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=60, max_features="sqrt",
                                    class_weight={0: 1, 1: spw}, n_jobs=-1, random_state=C.RANDOM).fit(Xtr, ytr)
        val_p["RandomForest"], test_p["RandomForest"] = rf.predict_proba(Xva)[:, 1], rf.predict_proba(Xte)[:, 1]

    # 6. thresholds (validation) and metrics (test)
    thr, _, _ = M.thresholds_from_validation(yva, val_p["GBM_tuned"])
    stats = {}
    for m in test_p:
        t, _, _ = M.thresholds_from_validation(yva, val_p[m])
        stats[m] = M.full_metrics(yte, test_p[m], t["f1"], t["recall_at_floor"])

    # 7. calibration
    cal, cal_type, briers = M.fit_calibrators(val_p["GBM_tuned"], yva, test_p["GBM_tuned"], yte)
    log(f"calibrator kept: {cal_type} | Brier {briers}")

    # 7b. The API compares CALIBRATED probabilities with the threshold, so thresholds for the
    # deployed model must be solved on calibrated validation probabilities (not raw ones).
    cal_val = M.apply_calibrator(cal, cal_type, val_p["GBM_tuned"])
    cal_test = M.apply_calibrator(cal, cal_type, test_p["GBM_tuned"])
    thr, _, _ = M.thresholds_from_validation(yva, cal_val)
    stats["GBM_tuned_calibrated"] = M.full_metrics(yte, cal_test, thr["f1"], thr["recall_at_floor"])
    log(f"deployment thresholds (calibrated scale): {thr}")

    # 8. robustness
    robust = {}
    if not args.skip_cv:
        log("grouped repeated CV ...")
        roc_ci, pr_ci = grouped_repeated_cv(Xtr, pd.Series(ytr), train[C.CLIENT].reset_index(drop=True),
                                            params, n_trees)
        robust["grouped_cv_roc"], robust["grouped_cv_pr"] = roc_ci, pr_ci
        log("out-of-time ...")
        robust["out_of_time"] = out_of_time(df, prep, params, n_trees)

    # 9. persist bundle + metrics
    joblib.dump({"prep": prep, "model": gbm, "calibrator": cal, "cal_type": cal_type,
                 "features": C.FEATURES, "threshold": float(thr["f1"]), "thresholds": thr,
                 "backend": M.backend(), "n_trees": int(n_trees), "best_params": params,
                 "grade_order": C.GRADE_ORD, "target": "early_delinquency (arrears_days>=1)"},
                C.BUNDLE_PATH)
    out = {"backend": M.backend(), "n_loans": int(len(df)), "n_clients": int(df[C.CLIENT].nunique()),
           "target_rate": float(df[C.TARGET].mean()), "thresholds": thr, "brier": briers,
           "calibrator": cal_type, "test_metrics": stats, "robustness": robust, "best_params": params}
    C.METRICS_PATH.write_text(json.dumps(out, indent=2, default=float))
    log(f"saved {C.BUNDLE_PATH.name} and {C.METRICS_PATH.name}")
    print(pd.DataFrame(stats).T.round(3).to_string())


if __name__ == "__main__":
    main()
