# ps-s5e12-diabetes

## Data

Kaggle competition data page:
https://www.kaggle.com/competitions/playground-series-s5e12/data


## Goal
Predict diabetes risk probability (tabular binary classification).



## Metric
ROC-AUC (main). Optionally evaluate calibration (Brier score / calibration curve).



## Validation
Stratified K-Fold CV with fixed random seed. Report mean ± std.



## How to run
```bash
pip install -r requirements.txt
# train
python -m src.train --config configs/logreg.yaml

# OOF metrics and visualization
python -m src.metrics --oof-path outputs/oof_logreg.csv --out-dir outputs/figures

# leakage checks (column-level)
python -m src.leakage --config configs/logreg.yaml --out-path outputs/leakage_report.csv

# ablation (group + single feature)
python -m src.ablation --config configs/logreg.yaml --mode both --out-path outputs/ablation_report.csv

# generate submisssion file
python -m src.predict --config configs/logreg.yaml --out submission.csv

```


## Results

### Baseline (logreg, no feature engineering)
- Validation: Stratified 5-fold CV (seed=42)
- Metric: ROC-AUC
- CV AUC: 0.6944 ± 0.0008 (mean ± std)
- OOF AUC: 0.6944
- OOF PR-AUC (Average Precision): 0.789777
- OOF Brier Score: 0.209097
- OOF KS: 0.274689

### CatBoost + feature engineering
- Validation: Stratified 5-fold CV (seed=42)
- Metric: ROC-AUC
- CV AUC: 0.7258306 ± 0.0006940 (mean ± std)
- OOF ROC-AUC: 0.725826
- OOF PR-AUC (Average Precision): 0.811523
- OOF Brier Score: 0.200482
- OOF KS: 0.325583

### LightGBM + feature engineering
- Validation: Stratified 5-fold CV (seed=42)
- Metric: ROC-AUC
- CV AUC: 0.724539 ± 0.000686 (mean ± std)
- OOF ROC-AUC: 0.724533
- OOF PR-AUC (Average Precision): 0.809857
- OOF Brier Score: 0.200835
- OOF KS: 0.324626

## One-page project brief (English + 日本語)

This project predicts diabetes risk probability from tabular clinical and lifestyle features in the Kaggle Playground Series S5E12 dataset. The objective is to support earlier intervention, targeted health programs, and better healthcare resource allocation by identifying high-risk individuals more effectively. The task is formulated as binary classification, with ROC-AUC as the primary metric and calibration-oriented measures such as Brier score used as secondary checks. The workflow relies on stratified 5-fold cross-validation with a fixed random seed, alongside leakage checks and ablation analysis to validate feature quality and model behavior. The baseline logistic regression model achieved a CV AUC of 0.6944, while feature-engineered CatBoost and LightGBM models improved performance to approximately 0.7258 and 0.7245, showing that richer feature design meaningfully improves predictive ranking quality.

本プロジェクトは、Kaggle Playground Series S5E12 の表形式データセットに含まれる臨床情報および生活習慣の特徴量を用いて、糖尿病リスク確率を予測することを目的としている。狙いは、高リスク者をより適切に特定することで、早期介入、対象を絞った健康施策、そして医療リソース配分の最適化を支援することである。問題設定は二値分類であり、主評価指標として ROC-AUC を採用し、補助的に Brier score などのキャリブレーション指標も確認する。検証では固定シード付きの Stratified 5-Fold CV を用い、あわせてリーク検査とアブレーション分析を実施することで、特徴量の妥当性とモデル挙動を確認している。ベースラインのロジスティック回帰は CV AUC 0.6944 を記録し、特徴量エンジニアリングを加えた CatBoost と LightGBM はそれぞれ約 0.7258 と 0.7245 まで改善しており、特徴量設計の強化が予測性能の向上に有効であることを示している。
