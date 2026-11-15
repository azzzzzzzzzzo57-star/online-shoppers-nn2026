# Online Shoppers Purchasing Intention — AIE231

## Week 9 — Data Engineer scope

**Problem:** Predict whether an online shopping session ends in a purchase (Revenue).

**Dataset:** UCI Online Shoppers Purchasing Intention, dataset ID 468: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention

This branch contains the Week 9 topic/checkpoint preparation and the Data Engineer work only: dataset download and inspection, duplicate/missing-value checks, cleaning, leakage-safe preprocessing, stratified splitting, and class-imbalance weights.

## Setup

Use Python 3.11 or 3.12.

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # macOS/Linux: source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

## Run from repository root

    python data/download_data.py
    python data/inspect_data.py
    python -m src.data

The data pipeline removes exact duplicate rows and rows with a missing target, normalizes the binary Revenue target, then creates stratified 70/15/15 train/validation/test splits with random seed 42. The numerical scaler and categorical encoder are fitted on training data only and then applied to validation and test data. Balanced class weights are calculated from the training labels.

The scripts save generated raw/processed data and preprocessing artifacts locally; these generated files are excluded from Git. Run the commands on the real UCI dataset to record the observed duplicate count and resulting split sizes.

## Project structure

- `data/download_data.py`: download the dataset from UCI
- `data/inspect_data.py`: inspect dimensions, types, missing values, duplicates and target distribution
- `data/clean_data.py`: report data-quality checks and save a local inspection copy
- `data/README.md`: dataset description and qualification criteria
- `src/data.py`: cleaning, stratified splitting, preprocessing and class weights
- `configs/base.yaml`: dataset and preprocessing configuration
- `notebooks/01_eda.ipynb`: exploratory data analysis
- `requirements.txt`: Python dependencies

Model architecture/training, evaluation, prediction, and later-stage experiment work are intentionally not included in this Week 9 Data Engineer branch.
