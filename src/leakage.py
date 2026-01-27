import argparse
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.features import prepare_features
from src.utils import ensure_dir, load_config, log, save_csv, seed_everything


SUSPICIOUS_NAME_PATTERNS: Tuple[str, ...] = (
    "target",
    "label",
    "outcome",
    "diagnos",
    "diabet",
    "y_",
)


def _has_suspicious_name(col: str) -> bool:
    col_l = col.lower()
    return any(pat in col_l for pat in SUSPICIOUS_NAME_PATTERNS)


def _encode_series_for_auc(s: pd.Series) -> np.ndarray:
    """Encode any dtype into a numeric vector suitable for a quick AUC check."""
    if s.dtype.kind in {"i", "u", "f", "b"}:
        arr = s.astype(float).values
    else:
        codes, _ = pd.factorize(s.astype(str), sort=True)
        arr = codes.astype(float)
    # Fill NA/inf with median-like fallback to keep roc_auc_score happy.
    arr = np.where(np.isfinite(arr), arr, np.nan)
    if np.isnan(arr).all():
        return np.zeros(len(arr), dtype=float)
    fill = float(np.nanmedian(arr))
    arr = np.where(np.isnan(arr), fill, arr)
    return arr


def _safe_auc(y: np.ndarray, x: np.ndarray) -> float:
    """Compute univariate AUC; return 0.5 when not computable."""
    try:
        # If the feature is constant, AUC is undefined; treat as 0.5.
        if np.allclose(x, x[0]):
            return 0.5
        return float(roc_auc_score(y, x))
    except Exception:
        return 0.5


def _safe_corr(y: np.ndarray, x: np.ndarray) -> float:
    try:
        if np.allclose(x, x[0]):
            return 0.0
        return float(np.corrcoef(y, x)[0, 1])
    except Exception:
        return 0.0


def build_leakage_report(
    X: pd.DataFrame,
    y: np.ndarray,
    auc_flag_threshold: float,
    corr_flag_threshold: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    y_float = y.astype(float)

    for col in X.columns:
        s = X[col]
        x = _encode_series_for_auc(s)

        auc = _safe_auc(y, x)
        auc_strength = abs(auc - 0.5)
        corr = _safe_corr(y_float, x)

        # Exact match (or inverted match) is a strong leakage signal.
        eq_rate = float(np.mean(x == y_float))
        inv_rate = float(np.mean(x == (1.0 - y_float)))

        suspicious_name = _has_suspicious_name(col)
        auc_flag = auc_strength >= (auc_flag_threshold - 0.5)
        corr_flag = abs(corr) >= corr_flag_threshold
        eq_flag = eq_rate >= 0.98 or inv_rate >= 0.98

        flags = []
        if suspicious_name:
            flags.append("name")
        if auc_flag:
            flags.append("auc")
        if corr_flag:
            flags.append("corr")
        if eq_flag:
            flags.append("eq")

        rows.append(
            {
                "feature": col,
                "dtype": str(s.dtype),
                "n_unique": int(s.nunique(dropna=False)),
                "missing_rate": float(s.isna().mean()),
                "auc": auc,
                "auc_strength": auc_strength,
                "corr_with_target": corr,
                "eq_rate": eq_rate,
                "inv_rate": inv_rate,
                "suspicious_name": bool(suspicious_name),
                "flags": ",".join(flags),
                "flagged": bool(flags),
            }
        )

    report = pd.DataFrame(rows)
    report = report.sort_values(
        by=["flagged", "auc_strength", "eq_rate", "inv_rate"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Column-level leakage checks.")
    parser.add_argument("--config", required=True, help="Path to config yaml (same as training).")
    parser.add_argument(
        "--out-path",
        default=os.path.join("outputs", "leakage_report.csv"),
        help="Where to save the leakage report CSV.",
    )
    parser.add_argument(
        "--auc-flag-threshold",
        type=float,
        default=0.95,
        help="Flag features whose univariate AUC is >= this value (or <= 1-this).",
    )
    parser.add_argument(
        "--corr-flag-threshold",
        type=float,
        default=0.98,
        help="Flag features whose absolute correlation with target exceeds this value.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    train_path = cfg["data"]["train_path"]
    target = cfg["data"]["target"]
    id_col = cfg["data"].get("id_col", None)
    feature_engineering = bool(cfg.get("feature_engineering", cfg["data"].get("feature_engineering", False)))

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

    ensure_dir(os.path.dirname(args.out_path) or "outputs")
    report = build_leakage_report(
        X=X,
        y=y,
        auc_flag_threshold=float(args.auc_flag_threshold),
        corr_flag_threshold=float(args.corr_flag_threshold),
    )
    save_csv(report, args.out_path, index=False)

    flagged = report[report["flagged"]]
    log(f"Leakage check complete. Flagged features: {len(flagged)} / {len(report)}")
    if not flagged.empty:
        preview_cols = ["feature", "auc", "corr_with_target", "eq_rate", "flags"]
        log("Top flagged features (preview):")
        for _, row in flagged.head(10)[preview_cols].iterrows():
            log(
                f"- {row['feature']}: auc={row['auc']:.4f}, "
                f"corr={row['corr_with_target']:.4f}, eq={row['eq_rate']:.3f}, flags={row['flags']}"
            )


if __name__ == "__main__":
    main()

