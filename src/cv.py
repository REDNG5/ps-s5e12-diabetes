from typing import Callable, Dict, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone


def build_stratified_kfold(n_splits: int, seed: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def run_cv(
    model,
    X,
    y,
    cv: StratifiedKFold,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
) -> Tuple[np.ndarray, Dict[str, float], list]:
    oof = np.zeros(len(y), dtype=float)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        fold_model = clone(model)
        fold_model.fit(X_tr, y_tr)
        pred = fold_model.predict_proba(X_va)[:, 1]
        oof[va_idx] = pred

        score = metric_fn(y_va, pred)
        fold_scores.append(score)

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))
    oof_score = float(metric_fn(y, oof))
    summary = {"mean": mean_score, "std": std_score, "oof": oof_score}

    return oof, summary, fold_scores
