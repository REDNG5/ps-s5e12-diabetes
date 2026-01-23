import argparse
import pandas as pd
import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config yaml (same as training)")
    parser.add_argument("--model-path", default="outputs/models/logreg.joblib")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    pack = joblib.load(args.model_path)
    clf = pack["pipeline"]
    cfg = pack["config"]

    test_path = cfg["data"]["test_path"]
    id_col = cfg["data"].get("id_col", "id")
    target = cfg["data"]["target"]

    test_df = pd.read_csv(test_path)

    if id_col not in test_df.columns:
        raise ValueError(f"ID column '{id_col}' not found in test.csv. Found: {list(test_df.columns)[:20]} ...")

    # Ensure we only use columns seen in training
    feat_cols = pack.get("feature_columns", None)
    if feat_cols is None:
        # fallback: drop id if present
        X_test = test_df.drop(columns=[id_col])
    else:
        X_test = test_df[feat_cols]

    pred = clf.predict_proba(X_test)[:, 1]

    # Kaggle submission usually needs: id + target
    sub = pd.DataFrame({id_col: test_df[id_col], target: pred})
    sub.to_csv(args.out, index=False)
    print(f"Saved submission to: {args.out}")


if __name__ == "__main__":
    main()
