"""Generate SYNTHETIC, random files shaped like the real datasets.

FOR SOFTWARE TESTING ONLY. These are random numbers, not patient data, and any
model trained on them says nothing about real disease prediction.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def _signal(rng, n, y, shift=1.0):
    return rng.normal(0, 1, n) + shift * y


def make(out_dir: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # heart (UCI-like: 13 features, target 0-4)
    n = 160
    y = rng.integers(0, 2, n)
    heart = pd.DataFrame({f"f{i}": _signal(rng, n, y, 0.8) for i in range(13)})
    heart["f1"] = rng.integers(0, 2, n)
    heart.loc[rng.choice(n, 4, replace=False), "f11"] = np.nan
    heart["target"] = np.where(y == 1, rng.integers(1, 5, n), 0)
    heart.to_csv(out_dir / "heart.csv", index=False)

    # diabetes (Pima-like, zeros = missing)
    n = 200
    y = rng.integers(0, 2, n)
    d = pd.DataFrame({c: np.abs(_signal(rng, n, y, 0.9)) * 30 + 1 for c in
                      ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                       "DiabetesPedigreeFunction", "Age"]})
    d.loc[rng.choice(n, 10, replace=False), "Insulin"] = 0
    d["Outcome"] = y
    d.to_csv(out_dir / "diabetes.csv", index=False)

    # kidney (text categoricals with tabs and '?')
    n = 160
    y = rng.integers(0, 2, n)
    k = pd.DataFrame({c: _signal(rng, n, y, 1.0) for c in
                      ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]})
    pick = lambda a, b: np.where(rng.random(n) < (0.2 + 0.6 * y), a, b)
    k["rbc"] = pick("abnormal", "normal"); k["pc"] = pick("abnormal", "normal")
    k["pcc"] = pick("present", "notpresent"); k["ba"] = pick("present", "notpresent")
    for c in ["htn", "dm", "cad", "pe", "ane"]:
        k[c] = pick("yes", "no")
    k["appet"] = pick("poor", "good")
    k["dm"] = k["dm"].replace({"yes": "\tyes", "no": " no"})  # messy whitespace
    k.loc[rng.choice(n, 8, replace=False), "rbc"] = "?"
    k.loc[rng.choice(n, 6, replace=False), "bgr"] = np.nan
    k["target"] = np.where(y == 1, "ckd", "notckd\t")
    k.to_csv(out_dir / "ckd.csv", index=False)

    # liver (ILPD-like; 1 = patient, 2 = not)
    n = 200
    y = rng.integers(0, 2, n)
    l = pd.DataFrame({c: np.abs(_signal(rng, n, y, 0.8)) * 5 + 1 for c in
                      ["Age", "TB", "DB", "Alkphos", "Sgpt", "Sgot", "TP", "ALB", "A/G Ratio"]})
    l.insert(1, "Gender", rng.choice(["Male", "Female"], n))
    l.loc[rng.choice(n, 3, replace=False), "A/G Ratio"] = np.nan
    l["target"] = np.where(y == 1, 1, 2)
    l.to_csv(out_dir / "ilpd.csv", index=False)


if __name__ == "__main__":
    make(Path(__file__).resolve().parent / "synthetic")
