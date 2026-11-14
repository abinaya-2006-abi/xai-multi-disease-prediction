# 🏥 Explainable AI (XAI) Multi-Disease Prediction System

An end-to-end, interpretable clinical decision support system that predicts four chronic diseases—**Heart Disease**, **Diabetes**, **Chronic Kidney Disease (CKD)**, and **Liver Disease**—under a single unified platform. 

The system combines a novel **LightGBM + CatBoost Stacking Ensemble** (with a Logistic Regression meta-learner) with **SHAP (SHapley Additive exPlanations)** to deliver both high predictive accuracy and transparent, patient-level clinical reasoning via an interactive **Streamlit** web application.

---

## 🌟 Key Features

* **Unified Multi-Disease Prediction:** Predicts risk across four major chronic conditions from a single interface.
* **Novel Stacking Ensemble Architecture:** Combines **LightGBM** (leaf-wise tree growth) and **CatBoost** (ordered boosting) as base learners with a **Logistic Regression** meta-learner, benchmarked against 12 baseline and ensemble models.
* **Clinical Transparency with SHAP:** Generates global **Summary Plots**, patient-specific **Force Plots**, and cumulative **Waterfall Plots** to explain *why* a prediction was made.
* **Robust Clinical Preprocessing:** Implements K-Nearest Neighbors (`KNNImputer`) for missing clinical values, `MinMaxScaler` normalization, and `SMOTE` oversampling on training splits to address class imbalance without data leakage.
* **Interactive Web Dashboard:** Built with **Streamlit** for real-time patient parameter input, risk probability scoring, and instant visual SHAP explanations.

---

## 📊 Datasets Used

All datasets are publicly available clinical benchmarks sourced from the UCI Machine Learning Repository and Kaggle:

| Disease Module | Dataset Name | Samples | Features | Target Variable |
| :--- | :--- | :---: | :---: | :--- |
| **Heart Disease** | UCI Heart Disease Dataset | 303 | 13 | `target` (0 = Healthy, 1 = Disease) |
| **Diabetes** | Pima Indians Diabetes Dataset | 768 | 8 | `Outcome` (0 = Negative, 1 = Positive) |
| **Kidney Disease** | UCI Chronic Kidney Disease (CKD) | 400 | 24 | `classification` (`ckd` = 1, `notckd` = 0) |
| **Liver Disease** | Indian Liver Patient Dataset (ILPD) | 583 | 10 | `Dataset` (1 = Liver Patient, 0 = Healthy) |

---

## 🏗️ System Architecture & Pipeline

1. **Data Cleaning & Imputation:** Biologically impossible zero values (e.g., Glucose, BMI in Diabetes) and malformed strings (in CKD) are converted to `NaN` and imputed using `KNNImputer (k=5)`.
2. **Encoding & Scaling:** Categorical clinical markers are encoded numerically, and continuous features are normalized to `[0, 1]` using `MinMaxScaler`.
3. **Train/Test Split & Balancing:** Stratified 80/20 split followed by `SMOTE` applied strictly to the training set.
4. **Model Benchmarking:** Evaluated across 13 classifiers (Logistic Regression, Decision Tree, KNN, SVM, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost, ExtraTrees+LightGBM Stack, LightGBM+CatBoost Voting, and the **Proposed LightGBM+CatBoost Stack**) using 10-fold stratified cross-validation.
5. **Explainability & Deployment:** Extracts feature attributions via `shap.TreeExplainer` and serves predictions through Streamlit.

---

## 📁 Project Structure

```text
xai-multi-disease-prediction/
│
├── app/                      # Streamlit web application files
│   └── app.py                # Main dashboard interface
├── data/                     # Raw and processed clinical datasets
├── models/                   # Saved trained models (.pkl) and scalers
├── notebooks/                # Jupyter notebooks for EDA, training & benchmarking
├── results/                  # Model comparison tables (.xlsx) and SHAP plots (.png/.html)
├── src/                      # Modular Python scripts for preprocessing and training
├── requirements.txt          # Python dependencies for reproducibility
└── README.md                 # Project documentation
