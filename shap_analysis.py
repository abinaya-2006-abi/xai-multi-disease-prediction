"""SHAP explanations for the trained proposed model (the whole stack, model-agnostic).

Usage:
    python src/shap_analysis.py --diseases heart kidney
Requires models/<disease>_proposed.joblib created by train_models.py.
Outputs in results/shap/<disease>/: beeswarm.png, bar.png, waterfall.png, force.html,
mean_abs_shap.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from common import DEFAULT_MODEL_DIR, DEFAULT_OUT_DIR, DISEASES, transform_features


def make_explainer(bundle):
    """Explainer over the stack's probability of the positive class, in the scaled feature space.

    Features are explained after imputation + scaling (what the model actually sees);
    plots are labelled with the original (unscaled) values for readability.
    """
    pipe = bundle["pipeline"]
    stack = pipe.named_steps["model"]
    _, bg_scaled = transform_features(pipe, bundle["X_background"])
    predict = lambda data: stack.predict_proba(pd.DataFrame(data, columns=bundle["features"]))[:, 1]
    algorithm = "exact" if len(bundle["features"]) <= 10 else "permutation"
    return shap.Explainer(predict, shap.maskers.Independent(bg_scaled.values), algorithm=algorithm,
                          feature_names=bundle["features"])


def explain(bundle, n_samples: int = 100, seed: int = 42):
    pipe = bundle["pipeline"]
    X_test = bundle["X_test"]
    sample = X_test.sample(min(n_samples, len(X_test)), random_state=seed)
    raw, scaled = transform_features(pipe, sample)
    explainer = make_explainer(bundle)
    kwargs = {"max_evals": 2 * len(bundle["features"]) + 1} if len(bundle["features"]) > 10 else {}
    exp = explainer(scaled.values, **kwargs)
    # show real-world (unscaled) values on the plots
    exp = shap.Explanation(values=exp.values, base_values=exp.base_values, data=raw.values,
                           feature_names=bundle["features"])
    return exp


def save_plots(exp, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    shap.plots.beeswarm(exp, show=False)
    plt.tight_layout(); plt.savefig(out / "beeswarm.png", dpi=300, bbox_inches="tight"); plt.close()
    shap.plots.bar(exp, show=False)
    plt.tight_layout(); plt.savefig(out / "bar.png", dpi=300, bbox_inches="tight"); plt.close()
    shap.plots.waterfall(exp[0], show=False)
    plt.tight_layout(); plt.savefig(out / "waterfall.png", dpi=300, bbox_inches="tight"); plt.close()
    fp = shap.force_plot(float(np.mean(exp.base_values)), exp.values[0], exp.data[0],
                         feature_names=list(exp.feature_names))
    shap.save_html(str(out / "force.html"), fp)
    pd.DataFrame({"feature": exp.feature_names, "mean_abs_shap": np.abs(exp.values).mean(0)}) \
        .sort_values("mean_abs_shap", ascending=False).to_csv(out / "mean_abs_shap.csv", index=False)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--diseases", nargs="+", default=DISEASES, choices=DISEASES)
    p.add_argument("--n-samples", type=int, default=100)
    args = p.parse_args(argv)
    for d in args.diseases:
        bundle = joblib.load(args.model_dir / f"{d}_proposed.joblib")
        print(f"SHAP for {d} ...")
        save_plots(explain(bundle, args.n_samples), args.out_dir / "shap" / d)
    print("Done.")


if __name__ == "__main__":
    main()
