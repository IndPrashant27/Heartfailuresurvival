# utils/metrics.py
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def compute_all(y_true, y_prob, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred)),
        'roc_auc': float(roc_auc_score(y_true, y_prob))
    }

def save_metrics(d, path):
    with open(path, 'w') as f:
        json.dump(d, f, indent=2)
