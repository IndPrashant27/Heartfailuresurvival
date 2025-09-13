
[![Live Demo](https://img.shields.io/badge/Streamlit-Live_App-brightgreen)](https://heart-failure-survival-pro-ewdnqvxcsymzzzuvvjtbjl.streamlit.app/)


# Heart Failure Survival – Pro Edition

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/stable/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosted%20Trees-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-style ML project:

- **From-scratch Logistic Regression** baseline (your theory signal ✅)
- Model **benchmarking** (custom logistic regression, sklearn logistic, Random Forest, XGBoost)
- **Cross-validation metrics** + persisted artifacts
- **Streamlit UI** with dark mode, modern cards, local images, and **SHAP** explainability
- Clean project layout, **Dockerfile**, and one-command run

> **Disclaimer**: This repository is for **educational** purposes only and is **not** for clinical use.

---

## ✨ Demo (Screenshots)

| App Header & Prediction | SHAP Explainability |
|--|--|
| ![header](assets/heart.jpg) | <img width="1470" height="956" alt="Screenshot 2025-08-23 at 12 06 59 PM" src="https://github.com/user-attachments/assets/5d3cc42e-db02-4f8f-9aef-ece15124e703" />|


> You can add real screenshots by pressing `s` in your browser or using macOS `⌘+Shift+4` and saving under `reports/figures/`.

---

## 🧰 Tech Stack

- **Python**, **NumPy**, **Pandas**
- **scikit-learn**, **XGBoost**
- **SHAP** for explainability
- **Streamlit** for the web UI
- **Matplotlib** for plots
- **joblib** for artifact persistence

---


---

## 🚀 Quickstart

```bash
# 1) Clone and enter
git clone https://github.com/<your-username>/heart-failure-survival-pro.git
cd heart-failure-survival-pro

# 2) (Recommended) Create & activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 3) Install requirements
pip install -r requirements.txt

# 4) Train models + export artifacts to models/artifacts/
python train.py

# 5) Launch the app
streamlit run app.py


