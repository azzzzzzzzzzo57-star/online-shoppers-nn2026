# Dataset documentation

## Source
- Dataset: Online Shoppers Purchasing Intention
- Repository: UCI Machine Learning Repository, dataset ID 468
- Source: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention
- Download: `python data/download_data.py`
- Local raw CSV: `data/raw/online_shoppers.csv`

## Structure and target
The published dataset contains 12,330 sessions, 17 input features, and the binary `Revenue` target (18 columns total). `Revenue=True` represents a completed purchase.

## Cleaning and missing values
The pipeline prints per-column missingness and the exact duplicate count. Exact duplicate rows are removed because they repeat the same complete session record. Rows with a missing target are excluded because a supervised target label is required. Missing numeric input values are imputed with the median and missing categorical input values with the most frequent category; these imputers are fitted on the training partition only.

## Splitting and leakage prevention
The data is split into stratified train/validation/test sets (70/15/15, seed 42). This is session-level tabular data, not time series, and no person/source grouping key is supplied. Imputers, StandardScaler, and OneHotEncoder are fitted on training data only and only transform validation/test data.

## Imbalance
The target is imbalanced. The actual class counts and percentages are calculated from the downloaded file and recorded in the EDA output and split summary. Balanced class weights are calculated from training labels only; the Week 10 validation ablation compares unweighted and weighted logistic regression.

## Reproducibility
Raw and processed datasets are generated locally and are not committed. Run the download script to obtain the dataset, and cite UCI in the final report.
