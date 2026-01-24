import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom_safe = denom.replace(0, np.nan)
    return numer / denom_safe


def add_basic_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    new_cols = []

    def _add(col_name: str, value: pd.Series) -> None:
        nonlocal df, new_cols
        df[col_name] = value
        new_cols.append(col_name)

    cols = df.columns

    if {"systolic_bp", "diastolic_bp"}.issubset(cols):
        pulse = df["systolic_bp"] - df["diastolic_bp"]
        _add("pulse_pressure", pulse)
        _add("mean_arterial_pressure", df["diastolic_bp"] + pulse / 3.0)

    if {"cholesterol_total", "hdl_cholesterol"}.issubset(cols):
        _add("chol_total_hdl_ratio", _safe_ratio(df["cholesterol_total"], df["hdl_cholesterol"]))

    if {"ldl_cholesterol", "hdl_cholesterol"}.issubset(cols):
        _add("ldl_hdl_ratio", _safe_ratio(df["ldl_cholesterol"], df["hdl_cholesterol"]))

    if {"triglycerides", "hdl_cholesterol"}.issubset(cols):
        _add("triglycerides_hdl_ratio", _safe_ratio(df["triglycerides"], df["hdl_cholesterol"]))

    if {"bmi", "waist_to_hip_ratio"}.issubset(cols):
        _add("bmi_waist_ratio", df["bmi"] * df["waist_to_hip_ratio"])

    if {"physical_activity_minutes_per_week", "sleep_hours_per_day"}.issubset(cols):
        sleep_hours_week = df["sleep_hours_per_day"] * 7.0
        _add("activity_per_sleep_hour", _safe_ratio(df["physical_activity_minutes_per_week"], sleep_hours_week))

    if {"physical_activity_minutes_per_week", "screen_time_hours_per_day"}.issubset(cols):
        screen_hours_week = df["screen_time_hours_per_day"] * 7.0
        _add("activity_per_screen_hour", _safe_ratio(df["physical_activity_minutes_per_week"], screen_hours_week))

    if {"age", "bmi"}.issubset(cols):
        _add("age_bmi_interaction", df["age"] * df["bmi"])

    return df, new_cols


def _save_target_distribution(df: pd.DataFrame, target: str, out_dir: str) -> None:
    if target not in df.columns:
        return
    counts = df[target].value_counts(dropna=False).sort_index()
    plt.figure(figsize=(5, 4))
    counts.plot(kind="bar")
    plt.title("Target Distribution")
    plt.xlabel(target)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "target_distribution.png"), dpi=150)
    plt.close()


def _save_numeric_histograms(df: pd.DataFrame, target: str, out_dir: str) -> None:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    numeric_cols = numeric_cols[:12]
    if not numeric_cols:
        return
    n = len(numeric_cols)
    rows = int(np.ceil(n / 3))
    plt.figure(figsize=(12, 3 * rows))
    for i, col in enumerate(numeric_cols, start=1):
        ax = plt.subplot(rows, 3, i)
        df[col].hist(bins=30, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "numeric_histograms.png"), dpi=150)
    plt.close()


def _save_correlation_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        return
    corr = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(shrink=0.8)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()


def _save_target_boxplots(df: pd.DataFrame, target: str, out_dir: str) -> None:
    if target not in df.columns:
        return
    candidates = ["age", "bmi", "systolic_bp", "cholesterol_total"]
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return
    rows = int(np.ceil(len(cols) / 2))
    plt.figure(figsize=(10, 4 * rows))
    for i, col in enumerate(cols, start=1):
        ax = plt.subplot(rows, 2, i)
        df.boxplot(column=col, by=target, ax=ax)
        ax.set_title(col)
        ax.set_xlabel(target)
        ax.set_ylabel("")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "target_boxplots.png"), dpi=150)
    plt.close()


def save_visualizations(df: pd.DataFrame, target: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    _save_target_distribution(df, target, out_dir)
    _save_numeric_histograms(df, target, out_dir)
    _save_correlation_heatmap(df, out_dir)
    _save_target_boxplots(df, target, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train.csv", help="Input CSV path")
    parser.add_argument("--out", default="outputs/train_fe.csv", help="Output CSV path")
    parser.add_argument("--plots-dir", default="outputs/figures", help="Directory to save plots")
    parser.add_argument("--target", default="diagnosed_diabetes", help="Target column name")
    parser.add_argument("--no-plots", action="store_true", help="Skip visualization output")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df, new_cols = add_basic_features(df)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved feature-engineered data to: {args.out}")
    if new_cols:
        print(f"Added features: {', '.join(new_cols)}")
    else:
        print("No new features added (required columns not found).")

    if not args.no_plots:
        save_visualizations(df, args.target, args.plots_dir)
        print(f"Saved plots to: {args.plots_dir}")


if __name__ == "__main__":
    main()
