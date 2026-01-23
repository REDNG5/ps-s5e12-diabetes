# ps-s5e12-diabetes



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