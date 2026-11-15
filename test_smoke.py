"""End-to-end smoke test on SYNTHETIC data (checks the code runs, not model quality)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import joblib
import pandas as pd
import pytest

import make_synthetic_data
import shap_analysis
import train_models
from common import DISEASES, load_disease
from models import get_models


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic")
    make_synthetic_data.make(d)
    return d


@pytest.mark.parametrize("disease", DISEASES)
def test_loaders(disease, data_dir):
    X, y = load_disease(disease, data_dir)
    assert set(y.unique()) == {0, 1}
    assert len(X) == len(y) and X.shape[1] >= 8
    assert all(str(t).startswith(("float", "Float")) for t in X.dtypes.astype(str))


def test_thirteen_models():
    assert len(get_models()) == 13


def test_train_and_shap(data_dir, tmp_path):
    out, mdl = tmp_path / "results", tmp_path / "models"
    train_models.main(["--data-dir", str(data_dir), "--out-dir", str(out), "--model-dir", str(mdl),
                       "--diseases", "heart", "kidney", "--cv-folds", "3"])
    table = pd.read_csv(out / "model_comparison_heart.csv")
    assert len(table) == 13
    for d in ("heart", "kidney"):
        shap_analysis.main(["--model-dir", str(mdl), "--out-dir", str(out), "--diseases", d, "--n-samples", "5"])
        assert (out / "shap" / d / "beeswarm.png").exists()
        assert (out / "shap" / d / "mean_abs_shap.csv").exists()
    assert joblib.load(mdl / "heart_proposed.joblib")["pipeline"] is not None
