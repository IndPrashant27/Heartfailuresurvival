# models/custom_logreg.py
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.05, n_iter=5000, reg_lambda=0.0, tol=1e-6, fit_intercept=True):
        self.lr = lr
        self.n_iter = n_iter
        self.reg_lambda = reg_lambda
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.w_ = None
        self.classes_ = np.array([0, 1])

    # sklearn API helpers
    def get_params(self, deep=True):
        return {
            "lr": self.lr,
            "n_iter": self.n_iter,
            "reg_lambda": self.reg_lambda,
            "tol": self.tol,
            "fit_intercept": self.fit_intercept,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        return np.c_[np.ones((X.shape[0], 1)), X]

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        Xb = self._add_intercept(X)
        n, d = Xb.shape
        self.w_ = np.zeros((d, 1))

        prev_loss = np.inf
        for _ in range(self.n_iter):
            logits = Xb @ self.w_
            p = self._sigmoid(logits)
            eps = 1e-12
            loss = (-1/n) * (y * np.log(p+eps) + (1-y) * np.log(1-p+eps)).sum() \
                   + (self.reg_lambda/(2*n)) * (self.w_[1:]**2).sum()
            grad = (1/n) * (Xb.T @ (p - y))
            grad[1:] += (self.reg_lambda/n) * self.w_[1:]
            self.w_ -= self.lr * grad
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        Xb = self._add_intercept(X)
        p = self._sigmoid(Xb @ self.w_)
        return np.hstack([1-p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
