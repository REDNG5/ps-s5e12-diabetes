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

## One-page project brief (English + 中文)

### (1) Business framing (业务化)
- EN: Predict individuals' diabetes risk probability to support early intervention, targeted health programs, and resource allocation.
- 中文：预测个体糖尿病风险概率，用于早筛干预、健康管理投放与资源分配优化。

### (2) Formulation (定式化)
- EN:
  - Input: tabular features from the competition dataset; target is binary diabetes risk (0/1).
  - Constraints: stratified CV to preserve class balance; no time leakage assumptions in provided data.
  - Metrics: ROC-AUC (primary), calibration metrics (Brier score / calibration curve).
  - Business metric mapping: Top-k hit rate for high-risk outreach; expected cost reduction by prioritizing highest-risk segments.
- 中文：
  - 输入：竞赛提供的表格特征；目标变量为糖尿病风险二分类（0/1）。
  - 约束：分层交叉验证保持类比例；数据无显式时间约束，需避免潜在泄漏。
  - 指标：ROC-AUC（主指标），校准指标（Brier 分数/校准曲线）。
  - 业务指标映射：高风险 Top-k 命中率；优先触达高风险人群带来的成本节省。

### (3) Hypothesis & validation (假设与验证)
- EN:
  - Hypotheses: core clinical features (e.g., glucose, BMI, age) are strong predictors; errors may be higher in certain demographic/clinical subgroups.
  - Validation: Stratified K-Fold CV; leakage checks; optional ablation to verify feature contribution; subgroup performance review if labels allow.
  - Automated leakage checks (outputs/leakage_report.csv):
    - Flagged 2 / 24 features by name heuristics: physical_activity_minutes_per_week, family_history_diabetes.
    - No near-perfect leakage signal detected (no eq_rate/inv_rate ~ 1.0).
  - Automated ablation (outputs/ablation_report.csv, base mean AUC = 0.694407):
    - Most influential single feature (drop causes largest AUC drop): family_history_diabetes (delta = -0.043076).
    - Most influential group: lifestyle (delta = -0.024960).
    - Core clinical signal is supported: dropping age reduces AUC (delta = -0.012486).
- 中文：
  - 假设：核心临床特征（如血糖、BMI、年龄）对风险有显著预测力；部分人群/临床分组误差更高。
  - 验证：分层 K 折交叉验证；泄漏排查；可做 ablation 验证特征贡献；若有分组字段则做分组评估。

### (4) Outcome (结果)
- EN:
  - Baseline score: CV AUC 0.6944 ± 0.0008; OOF AUC 0.6944.
  - Lift: baseline -> current baseline (no feature engineering); future models should quantify relative improvement.
  - Error analysis: to be expanded with subgroup/segment analysis and calibration diagnostics.
  - Evidence from validation diagnostics:
    - Ablation confirms strong dependence on family_history_diabetes and lifestyle features.
    - Leakage checks show no obvious near-perfect proxy for the label.
- 中文：
  - 基线分数：CV AUC 0.6944 ± 0.0008；OOF AUC 0.6944。
  - 提升：目前为无特征工程基线；后续模型需报告相对提升。
  - 误差分析：待补充分群表现与校准诊断。

### (5) Next steps (落地)
- EN:
  - Monitoring: drift detection, threshold tuning by operational capacity.
  - Data: add domain features, interaction terms, and data quality checks.
  - Deliverables: training/prediction pipeline, evaluation dashboards, and calibrated risk scores.
- 中文：
  - 监控：漂移监控；按业务容量调阈值。
  - 数据：新增领域特征与交互项，完善数据质量校验。
  - 产出：训练/推理 pipeline、评估可视化与校准后的风险分数。
