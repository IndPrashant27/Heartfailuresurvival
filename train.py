# train.py
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils.data import load_dataset, split_xy
from utils.metrics import compute_all, save_metrics
from models.custom_logreg import CustomLogisticRegression

DATA_PATH = 'data/heart_failure_clinical_records_dataset.csv'
ART_DIR = Path('models/artifacts'); ART_DIR.mkdir(parents=True, exist_ok=True)

# 1) Load
df = load_dataset(DATA_PATH)
X, y = split_xy(df)

# 2) Preprocess (all numeric -> standardize)
num_features = X.columns.tolist()
preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_features)
])

# 3) Models to compare
models = {
    'custom_logreg': Pipeline([
        ('prep', preprocess),
        ('clf', CustomLogisticRegression(lr=0.1, n_iter=8000, reg_lambda=1.0))
    ]),
    'sk_logreg': Pipeline([
        ('prep', preprocess),
        ('clf', LogisticRegression(max_iter=5000, C=1.0))
    ]),
    'random_forest': Pipeline([
        ('prep', preprocess),
        ('clf', RandomForestClassifier(n_estimators=400, max_depth=6, random_state=42))
    ]),
    'xgboost': Pipeline([
        ('prep', preprocess),
        ('clf', XGBClassifier(
            n_estimators=600, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, eval_metric='logloss', random_state=42
        ))
    ])
}

# 4) Cross-validated predictions + metrics
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics_all = {}

for name, pipe in models.items():
    # cross_val_predict on proba for ROC-AUC
    y_prob = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')[:, 1]
    # fit once on full data for predictions (threshold=0.5)
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    m = compute_all(y_true=y, y_prob=y_prob, y_pred=y_pred)
    metrics_all[name] = m

# 5) Pick best by ROC-AUC
best_name = max(metrics_all, key=lambda k: metrics_all[k]['roc_auc'])
best_pipe = models[best_name]

# Fit on full data and persist artifacts
best_pipe.fit(X, y)

# Save entire pipeline as one artifact
joblib.dump(best_pipe, ART_DIR / 'model_best.pkl')

# SHAP explainer (optional)
try:
    import shap
    clf = best_pipe.named_steps['clf']
    X_trans = best_pipe.named_steps['prep'].transform(X)
    if hasattr(clf, 'feature_importances_'):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_trans)
    else:
        explainer = shap.LinearExplainer(clf, X_trans, feature_perturbation='interventional')
        shap_values = explainer.shap_values(X_trans)
    joblib.dump({'explainer': explainer, 'feature_names': X.columns.tolist()}, ART_DIR / 'shap_explainer.pkl')
except Exception as e:
    print('SHAP setup skipped:', e)

# 6) Save metrics
save_metrics({'by_model': metrics_all, 'best_model': best_name}, ART_DIR / 'metrics.json')
print("Best model:", best_name)
print(json.dumps(metrics_all, indent=2))
