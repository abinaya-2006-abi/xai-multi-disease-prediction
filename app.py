"""Streamlit research prototype: predict + SHAP waterfall explanation.

Run:  streamlit run app/app.py     (after train_models.py has created models/*.joblib)
NOT a medical device and NOT medical advice.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from common import DEFAULT_MODEL_DIR, DISEASE_TITLES, DISEASES, transform_features
from shap_analysis import make_explainer

st.set_page_config(page_title="XAI Multi-Disease Prediction", layout="wide")
st.title("Explainable multi-disease prediction (research prototype)")
st.warning("Research prototype for academic demonstration only. It is not a medical device "
           "and must not be used for diagnosis or treatment decisions.")

available = [d for d in DISEASES if (DEFAULT_MODEL_DIR / f"{d}_proposed.joblib").exists()]
if not available:
    st.error("No trained models found. Run `python src/train_models.py` first.")
    st.stop()

disease = st.sidebar.selectbox("Disease", available, format_func=lambda d: DISEASE_TITLES[d])


@st.cache_resource
def load(d):
    b = joblib.load(DEFAULT_MODEL_DIR / f"{d}_proposed.joblib")
    return b, make_explainer(b)


bundle, explainer = load(disease)
st.subheader(f"{DISEASE_TITLES[disease]} - enter patient values")
values = {}
cols = st.columns(3)
for i, (name, info) in enumerate(bundle["feature_info"].items()):
    with cols[i % 3]:
        if info["discrete"]:
            opts = info["options"]
            values[name] = st.selectbox(name, opts, index=opts.index(min(opts, key=lambda v: abs(v - info["median"]))))
        else:
            values[name] = st.number_input(name, value=info["median"], min_value=info["min"] * 0.5 if info["min"] > 0 else info["min"] * 2 - 1,
                                           max_value=info["max"] * 2 + 1)

if st.button("Predict and explain"):
    row = pd.DataFrame([values], columns=bundle["features"])
    pipe = bundle["pipeline"]
    prob = float(pipe.predict_proba(row)[:, 1][0])
    st.metric("Predicted probability of disease", f"{prob:.1%}")
    raw, scaled = transform_features(pipe, row)
    exp = explainer(scaled.values)
    exp = shap.Explanation(values=exp.values, base_values=exp.base_values, data=raw.values,
                           feature_names=bundle["features"])
    shap.plots.waterfall(exp[0], show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption("Red bars push the prediction toward disease; blue bars push it away.")
