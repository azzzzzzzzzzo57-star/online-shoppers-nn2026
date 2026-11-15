# Data documentation

## Dataset
- Name: Online Shoppers Purchasing Intention
- Source: UCI Machine Learning Repository
- Dataset ID: 468
- Source page: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention
- Download command: python data/download_data.py
- Raw file: data/raw/online_shoppers.csv

## Dataset overview
The source dataset contains 12,330 sessions, 17 input features and the binary Revenue target (18 columns total).

## Data quality
The source snapshot has no missing values and contains exact duplicate records. The pipeline reports the duplicate count after download and removes exact duplicates before splitting. No imputation is needed for this source snapshot; unexpected missing target labels are excluded.

## Target distribution
The source snapshot is imbalanced: approximately 84.5% non-purchase sessions and 15.5% purchase sessions. Exact counts used in each experiment should be regenerated from the downloaded file and recorded in the EDA notebook.

## Reproducibility
Do not commit raw or processed data. Download the dataset locally with the script above. Cite UCI as the data source in the final report.
