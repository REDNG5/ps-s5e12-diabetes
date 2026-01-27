import argparse
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src import cv as cv_utils
from src.features import build_preprocess, prepare_features
from src.train import build_model
from src.utils import ensure_dir, load_config, log, save_csv, seed_everything


def _default_groups(columns: Iterable[str]) -> Dict[str, List[str]]:
    cols = set(columns)
    groups: Dict[str, List[str]] = {}

    core = [c for c in ["glucose", "bmi", "age"] if c in cols]
    if core:
        groups["core_clinical"] = core

    bp = [c for c in ["systolic_bp", "diastolic_bp"] if c in cols]
    if bp:
        groups["blood_pressure"] = bp

    lipids = [
        c
        for c in [
            "cholesterol_total",
            "hdl_cholesterol",
            "ldl_cholesterol",
            "triglycerides",
        ]
        if c in cols
    ]
    if lipids:
        groups["lipids"] = lipids

    lifestyle = [
        c
        for c in [
            "physical_activity_minutes_per_week",
            "sleep_hours_per_day",
            "screen_time_hours_per_day",
            "smoking",
            "alcohol_intake",
        ]
        if c in cols
    ]
    if lifestyle:
        groups["lifestyle"] = lifestyle

    return groups


def _parse_groups_arg(groups_arg: str, available_cols: Iterable[str]) -> Dict[str, List[str]]:
    """
    Parse a flat group definition string:
    "name1:colA,colB;name2:colC"
    """
    if not groups_arg:
        return {}
    cols = set(available_cols)
    groups: Dict[str, List[str]] = {}
    chunks = [c.strip() for c in groups_arg.split(";") if c.strip()]
    for chunk in chunks:
        if ":" not in chunk:
            continue
        name, col_str = chunk.split(":", 1)
        name = name.strip()
        members = [c.strip() for c in col_str.split(",") if c.strip()]
        members = [c for c in members if c in cols]
        if name and members:
            groups[name] = members
    return groups


def _evaluate_cv_auc(
    model_name: str,
    model_params: Dict[str, object],
    seed: int,
    n_splits: int,
    X: pd.DataFrame,
    y: np.ndarray,
) -> Tuple[Dict[str, float], List[float]]:
    preprocess = build_preprocess(X)
    model = build_model(model_name, model_params, seed)
    clf = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    cv = cv_utils.build_stratified_kfold(n_splits=n_splits, seed=seed)
    _, summary, fold_scores = cv_utils.run_cv(clf, X, y, cv)
    return summary, fold_scores


def _ablate_by_groups(
    groups: Dict[str, List[str]],
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    model_params: Dict[str, object],
    seed: int,
    n_splits: int,
    base_mean_auc: float,
) -> pd.DataFrame:
    rows = []
    for name, members in groups.items():
        keep_cols = [c for c in X.columns if c not in set(members)]
        if not keep_cols:
            continue
        summary, _ = _evaluate_cv_auc(
            model_name=model_name,
            model_params=model_params,
            seed=seed,
            n_splits=n_splits,
            X=X[keep_cols],
            y=y,
        )
        mean_auc = float(summary["mean"])
        rows.append(
            {
                "type": "group",
                "name": name,
                "dropped": ",".join(members),
                "n_dropped": len(members),
                "mean_auc": mean_auc,
                "delta_vs_base": mean_auc - base_mean_auc,
            }
        )

    res = pd.DataFrame(rows)
    if not res.empty:
        res = res.sort_values(by="delta_vs_base", ascending=True).reset_index(drop=True)
    return res


def _ablate_single_features(
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    model_params: Dict[str, object],
    seed: int,
    n_splits: int,
    base_mean_auc: float,
    max_features: int,
) -> pd.DataFrame:
    rows = []
    cols = list(X.columns)
    if max_features > 0:
        cols = cols[:max_features]

    for col in cols:
        keep_cols = [c for c in X.columns if c != col]
        if not keep_cols:
            continue
        summary, _ = _evaluate_cv_auc(
            model_name=model_name,
            model_params=model_params,
            seed=seed,
            n_splits=n_splits,
            X=X[keep_cols],
            y=y,
        )
        mean_auc = float(summary["mean"])
        rows.append(
            {
                "type": "single",
                "name": col,
                "dropped": col,
                "n_dropped": 1,
                "mean_auc": mean_auc,
                "delta_vs_base": mean_auc - base_mean_auc,
            }
        )

    res = pd.DataFrame(rows)
    if not res.empty:
        res = res.sort_values(by="delta_vs_base", ascending=True).reset_index(drop=True)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study: drop feature groups or single features.")
    parser.add_argument("--config", required=True, help="Path to config yaml (same as training).")
    parser.add_argument(
        "--mode",
        choices=["group", "single", "both"],
        default="both",
        help="Ablation mode.",
    )
    parser.add_argument(
        "--groups",
        default="",
        help="Optional extra groups: \"name1:colA,colB;name2:colC\"",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=0,
        help="Limit single-feature ablation to the first N columns (0 means all).",
    )
    parser.add_argument(
        "--out-path",
        default=os.path.join("outputs", "ablation_report.csv"),
        help="Where to save the ablation report CSV.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    n_splits = int(cfg.get("n_splits", 5))
    seed_everything(seed)

    train_path = cfg["data"]["train_path"]
    target = cfg["data"]["target"]
    id_col = cfg["data"].get("id_col", None)
    feature_engineering = bool(cfg.get("feature_engineering", cfg["data"].get("feature_engineering", False)))

    model_name = cfg["model"]["name"]
    model_params = dict(cfg["model"].get("params", {}))

    df = pd.read_csv(train_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in: {train_path}")

    X, y, _, new_cols = prepare_features(
        df,
        target=target,
        id_col=id_col,
        feature_engineering=feature_engineering,
    )
    if new_cols:
        log(f"Feature engineering enabled; added: {', '.join(new_cols)}")

    base_summary, base_folds = _evaluate_cv_auc(
        model_name=model_name,
        model_params=model_params,
        seed=seed,
        n_splits=n_splits,
        X=X,
        y=y,
    )
    base_mean_auc = float(base_summary["mean"])
    log(
        "Base CV AUC: "
        f"mean={base_summary['mean']:.6f}, std={base_summary['std']:.6f}, oof={base_summary['oof']:.6f}"
    )
    for i, score in enumerate(base_folds, start=1):
        log(f"[Base Fold {i}] AUC = {score:.6f}")

    reports: List[pd.DataFrame] = []

    if args.mode in {"group", "both"}:
        groups = _default_groups(X.columns)
        extra = _parse_groups_arg(args.groups, X.columns)
        # Extra groups can override defaults with the same name.
        groups.update(extra)
        if groups:
            log(f"Running group ablation on {len(groups)} groups.")
            reports.append(
                _ablate_by_groups(
                    groups=groups,
                    X=X,
                    y=y,
                    model_name=model_name,
                    model_params=model_params,
                    seed=seed,
                    n_splits=n_splits,
                    base_mean_auc=base_mean_auc,
                )
            )
        else:
            log("No valid groups found for group ablation.")

    if args.mode in {"single", "both"}:
        max_features = int(args.max_features)
        if max_features > 0:
            log(f"Running single-feature ablation on first {max_features} features.")
        else:
            log("Running single-feature ablation on all features.")
        reports.append(
            _ablate_single_features(
                X=X,
                y=y,
                model_name=model_name,
                model_params=model_params,
                seed=seed,
                n_splits=n_splits,
                base_mean_auc=base_mean_auc,
                max_features=max_features,
            )
        )

    ensure_dir(os.path.dirname(args.out_path) or "outputs")
    if reports:
        report = pd.concat([r for r in reports if not r.empty], ignore_index=True)
    else:
        report = pd.DataFrame()

    if report.empty:
        log("Ablation produced no rows; nothing to save.")
        return

    report.insert(0, "base_mean_auc", base_mean_auc)
    save_csv(report, args.out_path, index=False)

    log("Ablation complete. Top drops (largest negative delta):")
    preview = report.sort_values("delta_vs_base").head(10)
    for _, row in preview.iterrows():
        log(
            f"- [{row['type']}] {row['name']}: mean_auc={row['mean_auc']:.6f}, "
            f"delta={row['delta_vs_base']:.6f}"
        )


if __name__ == "__main__":
    main()

