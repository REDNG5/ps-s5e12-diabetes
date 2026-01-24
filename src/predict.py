import argparse
import pandas as pd

from src.features import prepare_features
from src.utils import load_config, load_model, log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config yaml (same as training)")
    parser.add_argument("--model-path", default="outputs/models/logreg.joblib")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    pack = load_model(args.model_path)
    clf = pack["pipeline"]
    cfg = pack["config"]

    test_path = cfg["data"]["test_path"]
    id_col = cfg["data"].get("id_col", "id")
    target = cfg["data"]["target"]

    test_df = pd.read_csv(test_path)
    if id_col not in test_df.columns:
        raise ValueError(f"ID column '{id_col}' not found in test.csv. Found: {list(test_df.columns)[:20]} ...")

    feature_engineering = bool(pack.get("feature_engineering", False))
    X_test, _, _, _ = prepare_features(
        test_df,
        target=None,
        id_col=id_col,
        feature_engineering=feature_engineering,
    )

    feat_cols = pack.get("feature_columns", None)
    if feat_cols is not None:
        X_test = X_test[feat_cols]

    pred = clf.predict_proba(X_test)[:, 1]
    sub = pd.DataFrame({id_col: test_df[id_col], target: pred})
    sub.to_csv(args.out, index=False)
    log(f"Saved submission to: {args.out}")


if __name__ == "__main__":
    main()
