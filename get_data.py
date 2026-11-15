"""Download the public datasets used in this project into data/raw/.

Heart Disease, Chronic Kidney Disease and ILPD (Indian Liver Patient Dataset)
come from the UCI Machine Learning Repository (CC BY 4.0) and are fetched with
the `ucimlrepo` package. The Pima Indians Diabetes dataset is NOT downloaded
automatically: see the instructions printed at the end of this script.

Usage:  python data/get_data.py
"""
from pathlib import Path

from ucimlrepo import fetch_ucirepo

RAW_DIR = Path(__file__).resolve().parent / "raw"

# name -> (UCI dataset id, output file)
UCI_DATASETS = {
    "heart": (45, "heart.csv"),
    "kidney": (336, "ckd.csv"),
    "liver": (225, "ilpd.csv"),
}

PIMA_HELP = """
Pima Indians Diabetes (not downloaded automatically)
  1. Open https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
  2. Check the licence shown on that page.
  3. Download diabetes.csv and save it as  data/raw/diabetes.csv
  (or, with the Kaggle CLI:  kaggle datasets download -d uciml/pima-indians-diabetes-database -p data/raw --unzip)
"""


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, (uci_id, filename) in UCI_DATASETS.items():
        dataset = fetch_ucirepo(id=uci_id)
        df = dataset.data.features.copy()
        # single target column, stored under the common name "target"
        df["target"] = dataset.data.targets.iloc[:, 0].values
        out = RAW_DIR / filename
        df.to_csv(out, index=False)
        print(f"[ok] {name:7s} -> {out}  ({len(df)} rows, {df.shape[1] - 1} features)")

    if not (RAW_DIR / "diabetes.csv").exists():
        print(PIMA_HELP)
    else:
        print("[ok] diabetes -> data/raw/diabetes.csv (already present)")


if __name__ == "__main__":
    main()
