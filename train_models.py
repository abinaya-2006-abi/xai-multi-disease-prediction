"""Train and evaluate the 13 models on each disease dataset.

Usage:
    python src/train_models.py --diseases heart diabetes kidney liver
    python src/train_models.py --diseases heart --tune

Outputs (nothing is shipped with the repo; they are created when YOU run it):
    results/model_comparison_<disease>.csv
    results/model_comparison_all.xlsx
    results/run_info.json
    models/<disease>_proposed.joblib
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

from common import (DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR, DEFAULT_OUT_DIR, DISEASES,
                    build_pipeline, feature_info, load_disease)
from models import PROPOSED, TUNING_GRID, get_models


def evaluate(pipe, X_tr, y_tr, X_te, y_te, cv) -> dict:
    cv_res = cross_validate(clone(pipe), X_tr, y_tr, cv=cv, scoring=["roc_auc", "f1"], n_jobs=1)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, 1]
    return {
        "Accuracy": 100 * accuracy_score(y_te, pred),
        "AUC-ROC": 100 * roc_auc_score(y_te, proba),
        "Precision": 100 * precision_score(y_te, pred, zero_division=0),
        "Recall": 100 * recall_score(y_te, pred, zero_division=0),
        "F1": 100 * f1_score(y_te, pred, zero_division=0),
        "CV AUC (mean)": 100 * cv_res["test_roc_auc"].mean(),
        "CV AUC (std)": 100 * cv_res["test_roc_auc"].std(),
        "CV F1 (mean)": 100 * cv_res["test_f1"].mean(),
    }


def run_disease(disease, args) -> pd.DataFrame:
    X, y = load_disease(disease, args.data_dir)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed)
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    print(f"\n=== {disease}: {len(X)} rows, {X.shape[1]} features, "
          f"positive rate {y.mean():.2%}; train {len(X_tr)} / test {len(X_te)}")

    rows, proposed_pipe = [], None
    for name, model in get_models(args.seed).items():
        t0 = time.time()
        pipe = build_pipeline(model, seed=args.seed)
        if name == PROPOSED and args.tune:
            search = GridSearchCV(pipe, TUNING_GRID, cv=args.tune_folds, scoring="roc_auc", n_jobs=1)
            search.fit(X_tr, y_tr)
            pipe = build_pipeline(model, seed=args.seed).set_params(**search.best_params_)
            print(f"   tuned {name}: {search.best_params_}")
        metrics = evaluate(pipe, X_tr, y_tr, X_te, y_te, cv)
        rows.append({"Model": name, **metrics})
        print(f"   {name:38s} acc={metrics['Accuracy']:.2f} auc={metrics['AUC-ROC']:.2f} "
              f"({time.time() - t0:.0f}s)")
        if name == PROPOSED:
            proposed_pipe = pipe

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).round(2)
    df.to_csv(args.out_dir / f"model_comparison_{disease}.csv", index=False)

    joblib.dump({
        "disease": disease, "pipeline": proposed_pipe, "features": list(X.columns),
        "feature_info": feature_info(X), "X_background": X_tr.sample(min(100, len(X_tr)), random_state=args.seed),
        "X_test": X_te, "y_test": y_te, "seed": args.seed, "model_name": PROPOSED,
    }, args.model_dir / f"{disease}_proposed.joblib")
    return df


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--diseases", nargs="+", default=DISEASES, choices=DISEASES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--tune", action="store_true", help="grid-search the proposed stack")
    p.add_argument("--tune-folds", type=int, default=3)
    args = p.parse_args(argv)

    tables = {d: run_disease(d, args) for d in args.diseases}

    with pd.ExcelWriter(args.out_dir / "model_comparison_all.xlsx") as xw:
        for d, df in tables.items():
            df.to_excel(xw, sheet_name=d, index=False)

    import catboost, lightgbm, shap, sklearn, xgboost
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "seed": args.seed, "cv_folds": args.cv_folds, "test_size": args.test_size,
            "tuned": args.tune, "diseases": args.diseases,
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                         "lightgbm": lightgbm.__version__, "catboost": catboost.__version__,
                         "xgboost": xgboost.__version__, "shap": shap.__version__}}
    (args.out_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nDone. Tables in {args.out_dir}, models in {args.model_dir}")


if __name__ == "__main__":
    main()
