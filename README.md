# Online Shoppers Purchasing Intention — AIE231

This repository contains the Week 9 data-engineering pipeline and Week 10 EDA, baselines, and first neural-network model for binary classification.

## Team Member	Role	Responsibilities
Ali	Tech Lead	Coordinates the project, reviews integration, and ensures the technical work follows the project requirements.
Aya	Data Engineer	Downloads, inspects, cleans, and preprocesses the dataset; prepares the stratified train/validation/test splits and . Baseline and
Analysis Engineer EDA and both baselines EDA notebook, baseline scripts Trains Model A and compares it with the baselines.
Shahd	Model Architect 
Architecture design, hand parameter counts,regularization models/ module with Models B and C Runs the architecture and regularization ablations, writes the 
GradientTape step.
Mohamed Experimentation Engineer Hyperparameter tuning and experiment tracking Experiment runner,logging, learning-curve plots Runs the learning-rate, batchsize and optimizer studies.
Nada Evaluation and Deployment Engineer Test-set evaluation, error analysis, inference pipeline,demo evaluate.py,predict.py, demo app Produces the confusion matrix, ROC and failure-case analysis for every model

## Team Members
| Team Member | GitHub Username |
|---|---|
| Aya | [@azzzzzzzzzzo57-star](https://github.com/azzzzzzzzzzo57-star) |
| Ali | [@alimoh555ali-boop](https://github.com/alimoh555ali-boop) |
| Mohamed | [@mohamedtttt](https://github.com/mohamedtttt) |
| Nada Hesham | [@nadahesham791-droid](https://github.com/nadahesham791-droid) |
| Shahd Mostafa | [@shahdmostafa368-mr](https://github.com/shahdmostafa368-mr) |


## Problem definition
Predict whether an online shopping session ends in a purchase using `Revenue` (`False` = 0, `True` = 1).

- **Task:** binary classification
- **Primary metric:** macro-F1, which gives equal importance to both classes under class imbalance
- **Secondary metrics:** per-class precision/recall/F1, purchase recall, ROC-AUC, accuracy, and confusion matrix
- **Dataset:** [UCI Online Shoppers Purchasing Intention, ID 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention)

## Week 9 — Data engineering
- Download and inspect the UCI dataset.
- Remove exact duplicate rows and rows with missing target labels; report missing feature values.
- Impute missing numeric/categorical features with median/most-frequent values fitted on training data only.
- Create stratified 70/15/15 train/validation/test partitions (seed 42).
- Fit imputers, `StandardScaler`, and `OneHotEncoder` on training data only; validation and test are transform-only.
- Calculate balanced class weights from training labels only.
- Save processed splits, fitted preprocessor, class weights, and split summary under `data/processed/` locally (not committed).

The dataset is session-level tabular data, not a time series, and does not provide a person/source grouping key. Stratified random splitting is therefore used. Sequence windows and image augmentation do not apply to this dataset.

## Week 10 — EDA, baselines, and Model A
- `notebooks/01_eda.ipynb`: dataset structure, data quality, target imbalance, numerical distributions, and correlations.
- `notebooks/02_baselines.ipynb`: compare the trivial baseline, learned baseline, and Model A using validation data.
- **Trivial baseline:** `DummyClassifier(strategy="most_frequent")`.
- **Learned baseline:** logistic regression.
- **Imbalance ablation:** compare unweighted and `class_weight="balanced"` logistic regression on the same validation split; report per-class recall.
- **Model A:** feed-forward neural network with 64- and 32-unit hidden layers, dropout, early stopping, and training-only class weights.
- Baselines and Model A use validation data for comparison; the test set is not used for model selection.
- `src/evaluate.py` selects the candidate with the best validation macro-F1 and evaluates the held-out test split once.

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

The Week 10 runner executes preprocessing, both baselines, then Model A, in that order. It does **not** load or evaluate the test split.

After model selection is complete, run the held-out test evaluation exactly once:

```bash
python -m src.evaluate
```

## Output files (generated locally)
- `data/processed/train_data.joblib`, `validation_data.joblib`, `test_data.joblib`
- `data/processed/preprocessor.joblib`, `class_weights.joblib`, `split_summary.json`
- `models/week10_logistic_unweighted.joblib`, `week10_logistic_balanced.joblib`, `week10_model_a.keras`
- `experiments/week10_baseline_validation.json`, `week10_model_a_validation.json`
- `experiments/final_test_evaluation.json` (only after the final test command)

Generated data, fitted models, and experiment outputs are intentionally excluded from Git. Run the documented commands to generate them locally.

## Project structure
- `configs/base.yaml` — project, preprocessing, and split settings
- `data/download_data.py` — download dataset from UCI
- `data/inspect_data.py` — inspect shape, types, missingness, duplicates, and target distribution
- `data/clean_data.py` — row-level quality checks and cleaned inspection copy
- `data/README.md` — dataset documentation
- `src/data.py` — cleaning, stratified split, train-only preprocessing, and class weights
- `src/baselines.py` — majority and logistic-regression baselines plus imbalance ablation
- `src/models.py` — Model A neural-network architecture
- `src/train.py` — train Model A and evaluate on validation data
- `src/run_week10.py` — reproducible preprocessing, baselines, and Model A runner
- `src/evaluate.py` — validation-based model selection and one-time final test evaluation
- `notebooks/01_eda.ipynb` — exploratory data analysis
- `notebooks/02_baselines.ipynb` — Week 10 baseline and Model A comparison
- `reports/week9_week10_report.md` — methodology and results-reporting template
- `requirements.txt` — pinned Python dependencies

## Dataset citation
Sakar, C. O., & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q
