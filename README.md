# Online Shoppers Purchasing Intention — AIE231

This repository contains the Week 9 data-engineering pipeline and Week 10 EDA/baseline checkpoint for a binary classification project.

## Problem definition
Predict whether an online shopping session ends in a purchase using the `Revenue` target (`False` = 0, `True` = 1). A false negative can miss a high-intent session; a false positive can waste marketing effort.

- **Task:** binary classification
- **Primary metric:** macro-F1 (gives equal importance to both classes under imbalance)
- **Secondary metrics:** purchase recall, per-class precision/recall/F1, ROC-AUC, accuracy, confusion matrix
- **Dataset:** [UCI Online Shoppers Purchasing Intention, ID 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention)

## Week 9 — Data engineering
- Download and inspect the UCI dataset.
- Remove exact duplicate rows and rows with missing target labels.
- Impute missing numeric/categorical features with median/most-frequent values fitted on training data only.
- Create stratified 70/15/15 train/validation/test partitions (seed 42).
- Fit imputers, `StandardScaler`, and `OneHotEncoder` on training data only; validation and test are transform-only.
- Calculate balanced class weights from training labels only.
- Save processed splits, preprocessor, split summary, and class weights under `data/processed/` locally (not committed).

The dataset is session-level tabular data, not a time series, and does not provide a person/source grouping key. Stratified random splitting is therefore used.

## Week 10 — EDA and baselines
- `notebooks/01_eda.ipynb`: dataset structure, data quality, target imbalance, numeric distributions, and correlations.
- `notebooks/02_baselines.ipynb`: validation baseline comparison and class-imbalance ablation.
- `src/baselines.py`: majority-class and logistic-regression baselines.
- Imbalance ablation: compare unweighted logistic regression with `class_weight="balanced"` and inspect per-class recall.
- Baseline scripts use validation data only; the test set is not loaded.
- `src/evaluate.py` is reserved for one final held-out test evaluation after model selection.

## Setup
Use Python 3.11 or 3.12.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Windows Git Bash: source .venv/Scripts/activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the workflow
From the repository root:

```bash
python data/download_data.py
python data/inspect_data.py
python data/clean_data.py
python -m src.run_week10
```

When all model choices are final, run the held-out test evaluation once:

```bash
python -m src.evaluate
```

Generated datasets, fitted models, and experiment outputs are intentionally excluded from Git. The scripts create them locally.

## Project structure
- `configs/base.yaml` — project, preprocessing, and split settings
- `data/download_data.py` — download dataset from UCI
- `data/inspect_data.py` — inspect shape, types, missingness, duplicates, and target distribution
- `data/clean_data.py` — row-level quality checks and cleaned inspection copy
- `data/README.md` — dataset documentation
- `src/data.py` — cleaning, stratified split, train-only preprocessing, and class weights
- `src/baselines.py` — majority and logistic-regression baselines plus imbalance ablation
- `src/run_week10.py` — reproducible preprocessing and validation baseline runner
- `src/evaluate.py` — one-time final test evaluation
- `notebooks/01_eda.ipynb` — exploratory data analysis
- `notebooks/02_baselines.ipynb` — Week 10 baseline analysis
- `reports/week9_week10_report.md` — report structure and methodology
- `requirements.txt` — pinned Python dependencies

## Dataset citation
Sakar, C. O., & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q
