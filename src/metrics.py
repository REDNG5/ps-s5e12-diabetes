import argparse
import os
from typing import Optional, Dict

import numpy as np
import pandas as pd
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
import joblib

from src.utils import ensure_dir, log
from src.features import prepare_features


def load_oof(oof_path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(oof_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {oof_path}.")
    if "oof_pred" not in df.columns:
        raise ValueError("Column 'oof_pred' not found in OOF file.")
    return df[[target, "oof_pred"]].dropna()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred)),
        "average_precision": float(average_precision_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_pred)),
    }


def plot_roc(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return roc_auc


def plot_pr(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return ap


def plot_calibration(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> float:
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.plot(prob_pred, prob_true, marker="o", label=f"Brier = {brier:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return brier


def plot_ks(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    ks = float(np.max(tpr - fpr))
    ks_idx = int(np.argmax(tpr - fpr))
    plt.figure(figsize=(5, 4))
    plt.plot(thresholds, tpr, label="TPR")
    plt.plot(thresholds, fpr, label="FPR")
    plt.axvline(thresholds[ks_idx], color="red", linestyle="--", label=f"KS = {ks:.4f}")
    plt.gca().invert_xaxis()
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("KS Curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return ks


def _get_feature_names(preprocess, input_columns):
    try:
        num_cols = preprocess.transformers_[0][2]
        cat_cols = preprocess.transformers_[1][2]
    except Exception:
        return list(input_columns)

    names = []
    names.extend(num_cols)
    try:
        ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
        names.extend(ohe_names)
    except Exception:
        names.extend(cat_cols)
    return names


def _is_tree_model(model) -> bool:
    name = model.__class__.__name__.lower()
    tree_markers = (
        "lgbm",
        "lightgbm",
        "catboost",
        "xgb",
        "randomforest",
        "gradientboost",
        "gbm",
        "extratrees",
        "decisiontree",
    )
    return any(marker in name for marker in tree_markers)


def plot_shap_summary(
    model_path: str,
    train_path: str,
    target: str,
    id_col: Optional[str],
    out_path: str,
    max_rows: int = 2000,
) -> bool:
    try:
        import shap  # noqa: WPS433
    except Exception:
        print("SHAP is not installed. Skip SHAP plots. Install with: pip install shap")
        return False

    pack = joblib.load(model_path)
    clf = pack["pipeline"]

    df = pd.read_csv(train_path)
    feature_engineering = bool(pack.get("feature_engineering", False))
    X, _, _, _ = prepare_features(
        df,
        target=target,
        id_col=id_col,
        feature_engineering=feature_engineering,
    )
    feat_cols = pack.get("feature_columns", None)
    if feat_cols is not None:
        X = X[feat_cols]
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=42)

    preprocess = clf.named_steps["preprocess"]
    model = clf.named_steps["model"]

    X_trans = preprocess.transform(X)
    feature_names = _get_feature_names(preprocess, X.columns)

    if _is_tree_model(model):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans)
    else:
        explainer = shap.LinearExplainer(model, X_trans)
        shap_values = explainer.shap_values(X_trans)

    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values = shap_values[1]

    shap.summary_plot(
        shap_values,
        X_trans,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-path", default="outputs/oof_logreg.csv")
    parser.add_argument("--target", default="diagnosed_diabetes")
    parser.add_argument("--out-dir", default="outputs/figures")
    parser.add_argument("--model-path", default="outputs/models/logreg.joblib")
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    oof_df = load_oof(args.oof_path, args.target)
    y_true = oof_df[args.target].astype(int).values
    y_pred = oof_df["oof_pred"].values

    roc_auc = plot_roc(y_true, y_pred, os.path.join(args.out_dir, "roc_curve.png"))
    ap = plot_pr(y_true, y_pred, os.path.join(args.out_dir, "pr_curve.png"))
    brier = plot_calibration(y_true, y_pred, os.path.join(args.out_dir, "calibration_curve.png"))
    ks = plot_ks(y_true, y_pred, os.path.join(args.out_dir, "ks_curve.png"))

    log(f"ROC AUC: {roc_auc:.6f}")
    log(f"Average Precision: {ap:.6f}")
    log(f"Brier Score: {brier:.6f}")
    log(f"KS: {ks:.6f}")

    if not args.skip_shap:
        out_path = os.path.join(args.out_dir, "shap_summary.png")
        plot_shap_summary(
            model_path=args.model_path,
            train_path=args.train_path,
            target=args.target,
            id_col=args.id_col,
            out_path=out_path,
        )


if __name__ == "__main__":
    main()
