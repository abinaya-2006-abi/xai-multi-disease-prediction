"""Shared helpers: dataset loading/cleaning and the leakage-safe preprocessing pipeline.

Every disease module is reduced to the same interface:
    X : DataFrame of numeric features (missing values left as NaN)
    y : Series of 0/1 labels (1 = disease present)

Imputation, scaling and SMOTE happen INSIDE an imblearn Pipeline, so they are
fitted on training data only (SMOTE is never applied to validation/test data).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "raw"
DEFAULT_OUT_DIR = ROOT / "results"
DEFAULT_MODEL_DIR = ROOT / "models"

DISEASES = ["heart", "diabetes", "kidney", "liver"]
DISEASE_TITLES = {
    "heart": "Heart disease",
    "diabetes": "Diabetes",
    "kidney": "Chronic kidney disease",
    "liver": "Liver disease",
}
DATA_FILES = {
    "heart": "heart.csv",
    "diabetes": "diabetes.csv",
    "kidney": "ckd.csv",
    "liver": "ilpd.csv",
}

# Text -> number maps for the CKD dataset (values are stripped/lower-cased first).
_CKD_CATEGORICAL = {
    "rbc": {"normal": 0, "abnormal": 1},
    "pc": {"normal": 0, "abnormal": 1},
    "pcc": {"notpresent": 0, "present": 1},
    "ba": {"notpresent": 0, "present": 1},
    "htn": {"no": 0, "yes": 1},
    "dm": {"no": 0, "yes": 1},
    "cad": {"no": 0, "yes": 1},
    "pe": {"no": 0, "yes": 1},
    "ane": {"no": 0, "yes": 1},
    "appet": {"good": 0, "poor": 1},
}
# In the Pima file a value of 0 in these columns is physiologically impossible -> missing.
_DIABETES_ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/get_data.py` to download the UCI datasets; "
            "diabetes.csv must be downloaded manually from Kaggle (see data/README.md)."
        )
    return pd.read_csv(path)


def _pop_target(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            return df.pop(name)
    raise KeyError(f"No target column found; expected one of {candidates}, got {list(df.columns)}")


def _clean_text(s: pd.Series) -> pd.Series:
    """Strip whitespace/tabs, lower-case, and turn '?'/empty strings into NaN."""
    out = s.astype(str).str.strip().str.lower()
    return out.replace({"?": np.nan, "": np.nan, "nan": np.nan, "none": np.nan})


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda c: pd.to_numeric(c.astype(str).str.strip(), errors="coerce"))


def _finish(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    keep = y.notna()
    X = X.loc[keep].reset_index(drop=True).astype(float)
    y = y.loc[keep].reset_index(drop=True).astype(int)
    if y.nunique() != 2:
        raise ValueError(f"Expected a binary target, found values {sorted(y.unique())}")
    return X, y


def load_disease(disease: str, data_dir: Path = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Load and clean one disease dataset. Returns (X, y) with y in {0, 1}."""
    if disease not in DATA_FILES:
        raise ValueError(f"Unknown disease '{disease}'. Choose from {DISEASES}")
    df = _read(Path(data_dir) / DATA_FILES[disease])

    if disease == "heart":
        raw = pd.to_numeric(_pop_target(df, ["target", "num"]), errors="coerce")
        y = (raw > 0).astype(float).where(raw.notna())  # UCI 'num' 1-4 -> 1
        return _finish(_numeric(df), y)

    if disease == "diabetes":
        y = pd.to_numeric(_pop_target(df, ["Outcome", "target"]), errors="coerce")
        X = _numeric(df)
        for col in _DIABETES_ZERO_AS_MISSING:
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)
        return _finish(X, y)

    if disease == "kidney":
        t = _clean_text(_pop_target(df, ["target", "class", "classification"]))
        y = t.map({"ckd": 1, "notckd": 0})
        df.columns = [c.strip() for c in df.columns]
        for col, mapping in _CKD_CATEGORICAL.items():
            if col in df.columns:
                df[col] = _clean_text(df[col]).map(mapping)
        return _finish(_numeric(df), y)

    # liver
    raw = pd.to_numeric(_pop_target(df, ["target", "Selector", "Dataset"]), errors="coerce")
    y = (raw == 1).astype(float).where(raw.notna())  # 1 = liver patient, 2 = not
    if "Gender" in df.columns:
        df["Gender"] = _clean_text(df["Gender"]).map({"male": 1, "female": 0})
    return _finish(_numeric(df), y)


def build_pipeline(model, seed: int = 42, use_smote: bool = True, knn_neighbors: int = 5) -> ImbPipeline:
    """KNN-impute -> MinMax scale -> SMOTE (train data only) -> model."""
    steps = [
        ("impute", KNNImputer(n_neighbors=knn_neighbors)),
        ("scale", MinMaxScaler()),
    ]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=seed)))
    steps.append(("model", model))
    return ImbPipeline(steps)


def transform_features(pipeline: ImbPipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a fitted pipeline's imputer and scaler (no SMOTE, no model).

    Returns (imputed_unscaled, imputed_scaled) as DataFrames with the original column names.
    """
    imputed = pipeline.named_steps["impute"].transform(X)
    scaled = pipeline.named_steps["scale"].transform(imputed)
    cols = list(X.columns)
    return pd.DataFrame(imputed, columns=cols, index=X.index), pd.DataFrame(scaled, columns=cols, index=X.index)


def feature_info(X: pd.DataFrame) -> dict:
    """Per-feature summary used by the Streamlit app to build its input widgets."""
    info = {}
    for col in X.columns:
        values = X[col].dropna()
        discrete = bool(len(values) and np.allclose(values, np.round(values)) and values.nunique() <= 10)
        info[col] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
            "discrete": discrete,
            "options": sorted(float(v) for v in values.unique()) if discrete else None,
        }
    return info
