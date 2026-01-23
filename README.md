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
python -m src.train --config configs/lgbm.yaml
python -m src.predict --config configs/lgbm.yaml --out submission.csv
```


## Results

### Baseline (Logistic Regression)
- Validation: Stratified 5-fold CV (seed=42)
- Metric: ROC-AUC
- CV AUC: 0.6944 ± 0.0008 (mean ± std)
- OOF AUC: 0.6944
