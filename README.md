# Explainable AI-Based Multi-Disease Prediction Using Ensemble Learning and SHAP

Code accompanying an IEEE conference paper. The repository contains code only: **no datasets, trained models
or results are included**; they are produced when you run the pipeline.

## What it does
- Predicts heart disease, diabetes, chronic kidney disease and liver disease from tabular clinical data.
- Compares 13 models (LR, DT, KNN, SVM, RF, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost,
  ExtraTrees+LightGBM stack, LightGBM+CatBoost voting) with a proposed **LightGBM + CatBoost stacking ensemble**
  (logistic-regression meta-learner).
- Leakage-safe preprocessing: KNN imputation -> MinMax scaling -> SMOTE (training folds only).
- Explains the full stacked model with SHAP (beeswarm, bar, waterfall, force plot, mean |SHAP| table).
- Includes a Streamlit demo (research prototype).

## Structure
```
data/get_data.py     download UCI datasets   | data/README.md  dataset sources, DOIs, licences
src/common.py        loading, cleaning, pipeline
src/models.py        the 13 models
src/train_models.py  train/evaluate, saves proposed model
src/shap_analysis.py SHAP plots
app/app.py           Streamlit demo
tests/               smoke test on synthetic data
notebooks/run_in_colab.ipynb
```

## Setup
```
python -m venv venv
venv\Scripts\activate        # Windows (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

## Usage
```
python data/get_data.py                 # heart, kidney, liver; then add data/raw/diabetes.csv manually
python src/train_models.py              # add --diseases heart kidney, --tune as needed
python src/shap_analysis.py
streamlit run app/app.py
pytest tests                            # synthetic-data smoke test
```
Outputs go to `results/` and `models/`. Seed 42, 80/20 stratified split, 5-fold CV.

## Data Availability (for the paper)
The datasets are public: Heart Disease (doi:10.24432/C52P4X), Chronic Kidney Disease (doi:10.24432/C5G020) and
ILPD (doi:10.24432/C5D02C) from the UCI Machine Learning Repository under CC BY 4.0, and the Pima Indians Diabetes
Database from Kaggle (originating from NIDDK). Code is available at
https://github.com/abinaya-2006-abi/xai-multi-disease-prediction. Data are not redistributed; `data/get_data.py`
and `data/README.md` describe how to obtain them.

## Limitations and disclaimer
Small public datasets; no external or clinical validation. This is a research prototype, **not a medical device**
and not for diagnosis or treatment.
