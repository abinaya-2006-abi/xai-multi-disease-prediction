"""The 13 models compared in the study, plus the proposed LightGBM + CatBoost stack."""
from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

PROPOSED = "LightGBM+CatBoost Stack (Proposed)"

# Small grid used only when `train_models.py --tune` is passed (tunes the proposed stack).
# Keys are pipeline parameter names: <pipeline step>__<stack estimator>__<parameter>.
TUNING_GRID = {
    "model__lgbm__n_estimators": [100, 200],
    "model__lgbm__learning_rate": [0.05, 0.1],
    "model__catboost__depth": [4, 6],
}


def _lgbm(seed: int) -> LGBMClassifier:
    return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                          random_state=seed, verbose=-1, n_jobs=1)


def _catboost(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, random_state=seed,
                              verbose=0, thread_count=1, allow_writing_files=False)


def _extra_trees(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=200, random_state=seed, n_jobs=1)


def get_models(seed: int = 42) -> dict:
    """Ordered dict name -> unfitted estimator (the order used in the paper's tables)."""
    stack_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
        "Extra Trees": _extra_trees(seed),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.05, eval_metric="logloss",
                                 random_state=seed, n_jobs=1, verbosity=0),
        "LightGBM": _lgbm(seed),
        "CatBoost": _catboost(seed),
        "ExtraTrees+LightGBM Stack": StackingClassifier(
            estimators=[("et", _extra_trees(seed)), ("lgbm", _lgbm(seed))],
            final_estimator=LogisticRegression(max_iter=1000), cv=stack_cv),
        "LightGBM+CatBoost Voting": VotingClassifier(
            estimators=[("lgbm", _lgbm(seed)), ("catboost", _catboost(seed))], voting="soft"),
        PROPOSED: StackingClassifier(
            estimators=[("lgbm", _lgbm(seed)), ("catboost", _catboost(seed))],
            final_estimator=LogisticRegression(max_iter=1000), cv=stack_cv),
    }
