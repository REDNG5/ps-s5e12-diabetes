import argparse
import os

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src import cv as cv_utils
from src.features import build_preprocess, prepare_features
from src.utils import load_config, log, save_csv, save_model, seed_everything, ensure_dir


def build_model(params: dict) -> LogisticRegression:
    class_weight = params.get("class_weight", None)
    if isinstance(class_weight, str) and class_weight.lower() == "null":
        class_weight = None

    return LogisticRegression(
        C=float(params.get("C", 1.0)),
        max_iter=int(params.get("max_iter", 2000)),
        solver=str(params.get("solver", "lbfgs")),
        class_weight=class_weight,
        n_jobs=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    n_splits = int(cfg.get("n_splits", 5))

    train_path = cfg["data"]["train_path"]
    target = cfg["data"]["target"]
    id_col = cfg["data"].get("id_col", None)
    feature_engineering = bool(cfg.get("feature_engineering", cfg["data"].get("feature_engineering", False)))

    seed_everything(seed)
    df = pd.read_csv(train_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in train.csv. Found: {list(df.columns)[:20]} ...")

    X, y, feature_columns, new_cols = prepare_features(
        df,
        target=target,
        id_col=id_col,
        feature_engineering=feature_engineering,
    )
    if new_cols:
        log(f"Added features: {', '.join(new_cols)}")

    preprocess = build_preprocess(X)
    model = build_model(cfg["model"]["params"])
    clf = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    cv = cv_utils.build_stratified_kfold(n_splits=n_splits, seed=seed)
    oof, summary, fold_scores = cv_utils.run_cv(clf, X, y, cv)

    for idx, score in enumerate(fold_scores, start=1):
        log(f"[Fold {idx}] AUC = {score:.6f}")
    log(f"CV AUC: mean={summary['mean']:.6f}, std={summary['std']:.6f}")
    log(f"OOF AUC: {summary['oof']:.6f}")

    ensure_dir("outputs")
    ensure_dir(os.path.join("outputs", "models"))

    clf.fit(X, y)
    model_path = os.path.join("outputs", "models", "logreg.joblib")
    save_model(
        {
            "pipeline": clf,
            "config": cfg,
            "feature_columns": feature_columns,
            "feature_engineering": feature_engineering,
        },
        model_path,
    )

    res = pd.DataFrame({"fold": np.arange(1, n_splits + 1), "auc": fold_scores})
    res.loc[len(res)] = ["mean", summary["mean"]]
    res.loc[len(res)] = ["std", summary["std"]]
    save_csv(res, os.path.join("outputs", "cv_results.csv"))

    oof_path = os.path.join("outputs", "oof_logreg.csv")
    if id_col and id_col in df.columns:
        oof_df = pd.DataFrame({id_col: df[id_col], target: y, "oof_pred": oof})
    else:
        oof_df = pd.DataFrame({target: y, "oof_pred": oof})
    save_csv(oof_df, oof_path)


if __name__ == "__main__":
    main()
