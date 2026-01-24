import os
import argparse
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )


def build_model(params: dict) -> LogisticRegression:
    # class_weight: null in yaml => Python None
    class_weight = params.get("class_weight", None)
    if isinstance(class_weight, str) and class_weight.lower() == "null":
        class_weight = None

    return LogisticRegression(
        C=float(params.get("C", 1.0)),
        max_iter=int(params.get("max_iter", 2000)),
        solver=str(params.get("solver", "lbfgs")),
        class_weight=class_weight,
        n_jobs=None,  # LogisticRegression for lbfgs doesn't use n_jobs
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    n_splits = int(cfg.get("n_splits", 5))

    train_path = cfg["data"]["train_path"]
    target = cfg["data"]["target"]
    id_col = cfg["data"].get("id_col", None)

    df = pd.read_csv(train_path)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in train.csv. Found: {list(df.columns)[:20]} ...")

    y = df[target].astype(int).values
    drop_cols = [target]
    if id_col and id_col in df.columns:
        drop_cols.append(id_col)

    X = df.drop(columns=drop_cols)

    preprocess = build_preprocess(X)
    model = build_model(cfg["model"]["params"])

    clf = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(df), dtype=float)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        clf.fit(X_tr, y_tr)
        pred = clf.predict_proba(X_va)[:, 1]
        oof[va_idx] = pred

        auc = roc_auc_score(y_va, pred)
        fold_scores.append(auc)
        print(f"[Fold {fold}] AUC = {auc:.6f}")

    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    oof_auc = roc_auc_score(y, oof)

    print(f"\nCV AUC: mean={mean_auc:.6f}, std={std_auc:.6f}")
    print(f"OOF AUC: {oof_auc:.6f}")

    # Fit on full data and save
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    clf.fit(X, y)
    model_path = "outputs/models/logreg.joblib"
    joblib.dump(
        {
            "pipeline": clf,
            "config": cfg,
            "feature_columns": X.columns.tolist(),
        },
        model_path,
    )
    print(f"\nSaved model to: {model_path}")

    # Save CV results
    res = pd.DataFrame({"fold": np.arange(1, n_splits + 1), "auc": fold_scores})
    res.loc[len(res)] = ["mean", mean_auc]
    res.loc[len(res)] = ["std", std_auc]
    res.to_csv("outputs/cv_results.csv", index=False)
    print("Saved CV results to: outputs/cv_results.csv")

    # Save OOF predictions for downstream analysis/plots
    oof_path = "outputs/oof_logreg.csv"
    if id_col and id_col in df.columns:
        oof_df = pd.DataFrame({id_col: df[id_col], target: y, "oof_pred": oof})
    else:
        oof_df = pd.DataFrame({target: y, "oof_pred": oof})
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved OOF predictions to: {oof_path}")


if __name__ == "__main__":
    main()
