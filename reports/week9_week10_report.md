# Week 9–10 Progress Report

## Project definition
Predict whether an online shopping session ends in a purchase using the binary target `Revenue` (False = no purchase, True = purchase).

- **Task:** binary classification
- **Primary metric:** macro-F1, because it weights both classes equally under class imbalance.
- **Secondary metrics:** purchase recall, per-class precision/recall/F1, ROC-AUC, accuracy, and confusion matrix.
- **Dataset:** [UCI Online Shoppers Purchasing Intention, ID 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention).

## Week 9 — Data engineering
The pipeline reports missing values and exact duplicates, removes exact duplicate rows, and removes rows with missing target labels. Missing numeric feature values are imputed with the training-set median; missing categorical feature values are imputed with the training-set most frequent value. StandardScaler and OneHotEncoder are also fitted on the training partition only.

The data is split into stratified train/validation/test partitions (70/15/15, random seed 42). This dataset is session-level tabular data and does not provide a person/source grouping key or a chronological sequence key for a forecasting task, so stratified random splitting is used. The held-out test set is not used for training or model selection.

Class imbalance is addressed with balanced class weights computed from training labels only. The validation ablation compares logistic regression without class weights and with `class_weight="balanced"`, reporting per-class recall.

## Week 10 — EDA and baselines
The EDA notebook reviews dimensions, data types, missingness, duplicates, target distribution, numerical distributions, and numeric correlations.

Two required baselines are evaluated on validation data:
1. **Trivial baseline:** `DummyClassifier(strategy="most_frequent")`.
2. **Learned baseline:** logistic regression.

The unweighted and class-weighted logistic regression comparison is the class-imbalance ablation. Baseline metrics are generated from the downloaded dataset and saved locally to `experiments/week10_baseline_validation.json`. Numerical results must be copied from an actual run; this report intentionally does not invent metric values.

## Reproduction
From the repository root:

```bash
python -m pip install -r requirements.txt
python data/download_data.py
python data/inspect_data.py
python data/clean_data.py
python -m src.run_week10
```

After model selection is complete, run the held-out test evaluation once:

```bash
python -m src.evaluate
```

## Results to record
Use the generated validation JSON to report macro-F1 and both class recalls for each baseline. Explain whether logistic regression improves over the majority classifier, and discuss the trade-off introduced by class weighting. Add final test metrics only after the one-time final evaluation.
